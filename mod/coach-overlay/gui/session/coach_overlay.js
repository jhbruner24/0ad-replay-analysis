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
	var names = model.names, mu = model.mu, sd = model.sd;
	var z = new Array(names.length);
	for (var i = 0; i < names.length; ++i) {
		var v = feats[names[i]];
		z[i] = ((v === undefined ? 0 : v) - mu[i]) / sd[i];
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

/* ---------- features: mirrors live_coach._live_features ---------- */

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

function CoachOverlay_Last(series) {
	if (!Array.isArray(series)) return null;
	for (var i = series.length - 1; i >= 0; --i)
		if (typeof series[i] === "number") return series[i];
	return null;
}

function CoachOverlay_Features(model, tSeconds, mine, them) {
	// Command features are not available live and rating is deliberately
	// absent (position-only model); both stay at their zero default.
	var feats = { "t_log": Math.log(tSeconds + 1), "rating_diff": 0, "rated": 0 };
	var mineSeq = CoachOverlay_FlattenSequences(mine.sequences);
	var themSeq = CoachOverlay_FlattenSequences(them.sequences);
	for (var i = 0; i < model.names.length; ++i) {
		var name = model.names[i];
		if (name.indexOf("diff.") !== 0) continue;
		var stat = name.substring(5);
		var my = CoachOverlay_Last(mineSeq[stat]);
		if (my === null) continue;
		var their = CoachOverlay_Last(themSeq[stat]);
		feats[name] = my - (their === null ? 0 : their);
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
		var feats = CoachOverlay_Features(g_CoachOverlay_Model, tSeconds,
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
