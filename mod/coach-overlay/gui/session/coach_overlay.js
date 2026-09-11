/**
 * Coach Overlay — draws a live win-probability readout in the top panel.
 *
 * Observer only: reads GetExtendedSimulationState via the GUI interface (the
 * same call the summary screen makes), scores it with the model exported by
 * export_model.py (gui/coach-overlay/model.json), and never touches the sim.
 *
 * It also publishes the raw state to saves/campaigns/coach-overlay/live_state.json
 * for external tools (live_coach.py); the panel does not depend on that.
 */

var g_CoachOverlay_IntervalMs = 5000;
var g_CoachOverlay_ModelPath = "gui/coach-overlay/model.json";
// GUI-context WriteJSONFile is path-restricted; saves/campaigns/ is one of the
// writable prefixes. Real path: <userdata>/saves/campaigns/coach-overlay/
var g_CoachOverlay_OutPath = "saves/campaigns/coach-overlay/live_state.json";
var g_CoachOverlay_Started = false;
var g_CoachOverlay_TickCount = 0;
var g_CoachOverlay_Model = null;

function CoachOverlay_IsReady() {
	// Session globals populate over the first few frames; wait for them.
	return typeof g_SimState !== "undefined" && !!g_SimState
		&& g_SimState.timeElapsed !== undefined
		&& typeof g_PlayerAssignments !== "undefined";
}

/* ---------- model: mirrors oadrep.model.predict_from_export ---------- */

function CoachOverlay_Interp(xs, ys, d) {
	if (d <= xs[0]) return ys[0];
	var n = xs.length;
	if (d >= xs[n - 1]) return ys[n - 1];
	// bisect_right(xs, d) - 1
	var lo = 0, hi = n;
	while (lo < hi) {
		var mid = (lo + hi) >> 1;
		if (xs[mid] <= d) lo = mid + 1; else hi = mid;
	}
	var i = lo - 1;
	var x0 = xs[i], x1 = xs[i + 1], y0 = ys[i], y1 = ys[i + 1];
	return x1 == x0 ? y0 : y0 + (y1 - y0) * (d - x0) / (x1 - x0);
}

function CoachOverlay_Predict(model, feats) {
	var names = model.names, mu = model.mu, sd = model.sd, lo = model.lo, hi = model.hi;
	var z = new Array(names.length);
	for (var i = 0; i < names.length; ++i) {
		var v = feats[names[i]];
		if (v === undefined) v = 0;
		if (lo) v = Math.min(Math.max(v, lo[i]), hi[i]);
		z[i] = (v - mu[i]) / sd[i];
	}
	var total = 0;
	for (var m = 0; m < model.members.length; ++m) {
		var mem = model.members[m];
		var d = mem.intercept;
		for (var j = 0; j < z.length; ++j) d += mem.coef[j] * z[j];
		total += CoachOverlay_Interp(mem.iso_x, mem.iso_y, d);
	}
	return total / model.members.length;
}

/* ---------- features: mirrors oadrep.model.live_features ---------- */

function CoachOverlay_FlattenSequences(raw) {
	var out = {};
	if (!raw || typeof raw !== "object") return out;
	for (var stat in raw) {
		if (stat == "time") continue;
		var series = raw[stat];
		if (Array.isArray(series))
			out[stat] = series;
		else if (series && typeof series === "object")
			for (var sub in series)
				if (Array.isArray(series[sub]))
					out[stat + "." + sub] = series[sub];
	}
	return out;
}

// model._value_at: value at or just before t (times ascending).
function CoachOverlay_ValueAt(times, values, t) {
	var last = null;
	for (var i = 0; i < times.length && i < values.length; ++i) {
		if (times[i] > t) break;
		if (typeof values[i] === "number") last = values[i];
	}
	return last;
}

// model.spec_value
function CoachOverlay_SpecValue(times, seqs, terms, window, t) {
	function totalAt(tt) {
		var total = 0, seen = false;
		for (var i = 0; i < terms.length; ++i) {
			var series = seqs[terms[i][0]];
			if (!series || !series.length) continue;
			var v = CoachOverlay_ValueAt(times, series, tt);
			if (v === null) continue;
			seen = true;
			total += terms[i][1] * v;
		}
		return seen ? total : null;
	}
	var now = totalAt(t);
	if (now === null) return null;
	if (window === null || window === undefined) return now;
	var before = totalAt(t - window);
	return (now - (before === null ? 0 : before)) / window;
}

// model.UNIT_COST — keep in sync.
var g_CoachOverlay_UnitCost = { "Worker": 110, "Cavalry": 180, "Champion": 260,
	"Siege": 380, "Ship": 220, "Hero": 600, "Trader": 180 };

// model._live_lookup
function CoachOverlay_LiveLookup(player, key) {
	if (key == "army_value") {
		var counts = player.classCounts;
		if (!counts || typeof counts !== "object") return null;
		var total = 0;
		for (var c in g_CoachOverlay_UnitCost)
			total += g_CoachOverlay_UnitCost[c] * (counts[c] || 0);
		return total;
	}
	var cur = player;
	var parts = key.split(".");
	for (var i = 0; i < parts.length; ++i) {
		if (!cur || typeof cur !== "object") return null;
		cur = cur[parts[i]];
	}
	if (cur === undefined || cur === null)
		return key.indexOf("classCounts.") === 0 ? 0 : null;
	return typeof cur === "number" ? cur : null;
}

// model.live_features: rating and command features are absent live and
// stay at their zero default.
function CoachOverlay_Features(model, mine, them) {
	var mineSeq = CoachOverlay_FlattenSequences(mine.sequences);
	var themSeq = CoachOverlay_FlattenSequences(them.sequences);
	var mineTimes = (mine.sequences && mine.sequences.time) || [];
	var themTimes = (them.sequences && them.sequences.time) || [];
	var t = mineTimes.length ? mineTimes[mineTimes.length - 1] : 0;
	var feats = { "t_log": Math.log(t + 1) };
	for (var name in model.spec) {
		var e = model.spec[name];
		var my = null, their = null;
		if (e.live) {
			my = CoachOverlay_LiveLookup(mine, e.live);
			their = CoachOverlay_LiveLookup(them, e.live);
		}
		if (my === null) {
			my = CoachOverlay_SpecValue(mineTimes, mineSeq, e.terms, e.window, t);
			their = CoachOverlay_SpecValue(themTimes, themSeq, e.terms, e.window, t);
		}
		if (my === null) continue;
		feats["diff." + name] = my - (their === null ? 0 : their);
	}
	return feats;
}

/**
 * Whose side to score. The viewed player if that is a real player, else
 * player 1. Returns null unless exactly one other (non-Gaia) player exists.
 */
function CoachOverlay_Perspective(players) {
	var viewed = (typeof g_ViewedPlayer !== "undefined" && g_ViewedPlayer > 0) ? g_ViewedPlayer : 1;
	var others = [];
	for (var i = 1; i < players.length; ++i)
		if (i != viewed && players[i]) others.push(i);
	if (!players[viewed] || others.length != 1) return null;
	return { "me": viewed, "them": others[0] };
}

/* ---------- drawing ---------- */

function CoachOverlay_Draw(text, p) {
	var textObj = Engine.GetGUIObjectByName("coachWinProbText");
	var bar = Engine.GetGUIObjectByName("coachWinProbBar");
	var bg = Engine.GetGUIObjectByName("coachWinProbBarBg");
	if (!textObj || !bar || !bg) return;
	textObj.caption = text;
	if (p === null) {
		bar.hidden = true;
		return;
	}
	bar.hidden = false;
	var cs = bg.getComputedSize();
	var width = (cs.right - cs.left) - 2;
	var size = bar.size;
	size.right = size.left + Math.max(1, Math.round(width * p));
	bar.size = size;
	bar.sprite = p >= 0.5 ? "color: 90 200 90 230" : "color: 220 80 80 230";
}

/* ---------- main loop ---------- */

function CoachOverlay_Tick() {
	try {
		var extState = Engine.GuiInterfaceCall("GetExtendedSimulationState");
		if (!extState) return;
		var tSeconds = extState.timeElapsed / 1000.0;
		var players = extState.players || [];

		Engine.WriteJSONFile(g_CoachOverlay_OutPath, {
			"schema": 1,
			"t_seconds": tSeconds,
			"written_at_ms": Date.now(),
			"tick": g_CoachOverlay_TickCount,
			"players": players,
			"map_name": (g_InitAttributes && g_InitAttributes.settings
				&& g_InitAttributes.settings.mapName) || null,
			"viewed_player_id": (typeof g_ViewedPlayer !== "undefined") ? g_ViewedPlayer : null,
		});
		g_CoachOverlay_TickCount += 1;

		if (!g_CoachOverlay_Model) {
			CoachOverlay_Draw("coach: no model", null);
			return;
		}
		var persp = CoachOverlay_Perspective(players);
		if (!persp) {
			CoachOverlay_Draw("coach: 1v1 only", null);
			return;
		}
		var feats = CoachOverlay_Features(g_CoachOverlay_Model,
			players[persp.me], players[persp.them]);
		var p = CoachOverlay_Predict(g_CoachOverlay_Model, feats);
		CoachOverlay_Draw("Win " + Math.round(p * 100) + "%", p);
	} catch (e) {
		warn("[coach-overlay] tick failed: " + e);
	}
}

function CoachOverlay_Loop() {
	if (!CoachOverlay_IsReady()) {
		setTimeout(CoachOverlay_Loop, 1000);
		return;
	}
	CoachOverlay_Tick();
	setTimeout(CoachOverlay_Loop, g_CoachOverlay_IntervalMs);
}

function CoachOverlay_Boot() {
	if (g_CoachOverlay_Started) return;
	g_CoachOverlay_Started = true;
	try {
		g_CoachOverlay_Model = Engine.ReadJSONFile(g_CoachOverlay_ModelPath);
	} catch (e) {
		warn("[coach-overlay] model load failed: " + e);
	}
	// Small initial delay so session.js has finished its first init pass.
	setTimeout(CoachOverlay_Loop, 500);
}

// The session GUI loads every JS file in gui/session/. Kicking off the loop
// at module load is safe: setTimeout defers work until the event loop runs.
if (typeof Engine !== "undefined")
	CoachOverlay_Boot();
