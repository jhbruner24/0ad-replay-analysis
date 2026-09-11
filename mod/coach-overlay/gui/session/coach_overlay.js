/**
 * Coach Overlay — publishes live sim state so an external tool can score P(win).
 *
 * Observer only: reads GetExtendedSimulationState via the GUI interface (the
 * same call the summary screen makes) and writes it to saves/campaigns/coach-overlay/live_state.json
 * under the user data directory. Never mutates simulation state.
 */

var g_CoachOverlay_IntervalMs = 5000;
// GUI-context WriteJSONFile is path-restricted; saves/campaigns/ is one of the
// writable prefixes (CampaignRun.js uses it). Real path: <userdata>/saves/campaigns/coach-overlay/
var g_CoachOverlay_OutPath = "saves/campaigns/coach-overlay/live_state.json";
// Written back by live_coach.py. Must always be exactly this many bytes: the
// engine's VFS caches a file's size on first sight and reads that many bytes
// on every later load (lib/file/vfs/vfs.cpp). We create it at this size here;
// the Python side overwrites it space-padded to the same size.
var g_CoachOverlay_InPath = "saves/campaigns/coach-overlay/p_win.json";
var g_CoachOverlay_InBytes = 512;
var g_CoachOverlay_StaleMs = 20000;
var g_CoachOverlay_Started = false;
var g_CoachOverlay_TickCount = 0;

function CoachOverlay_IsReady() {
	// Session globals populate over the first few frames; wait for them.
	return typeof g_SimState !== "undefined" && !!g_SimState
		&& g_SimState.timeElapsed !== undefined
		&& typeof g_PlayerAssignments !== "undefined";
}

function CoachOverlay_Publish() {
	try {
		var extState = Engine.GuiInterfaceCall("GetExtendedSimulationState");
		if (!extState) return;
		var payload = {
			"schema": 1,
			"t_seconds": extState.timeElapsed / 1000.0,
			"written_at_ms": Date.now(),
			"tick": g_CoachOverlay_TickCount,
			"players": extState.players,
			// InitAttributes lives on the sim in v0.28; harmless if absent.
			"map_name": (g_InitAttributes && g_InitAttributes.settings
				&& g_InitAttributes.settings.mapName) || null,
			"viewed_player_id": (typeof g_ViewedPlayer !== "undefined")
				? g_ViewedPlayer : null,
		};
		Engine.WriteJSONFile(g_CoachOverlay_OutPath, payload);
		g_CoachOverlay_TickCount += 1;
	} catch (e) {
		warn("[coach-overlay] publish failed: " + e);
	}
}

/**
 * Create p_win.json at its fixed size so the VFS registers the right length.
 * Also wipes any value left over from a previous game.
 */
function CoachOverlay_InitInbox() {
	try {
		var base = { "schema": 1, "p": null, "pad": "" };
		var len = JSON.stringify(base).length;
		base.pad = " ".repeat(g_CoachOverlay_InBytes - len);
		Engine.WriteJSONFile(g_CoachOverlay_InPath, base);
	} catch (e) {
		warn("[coach-overlay] inbox init failed: " + e);
	}
}

function CoachOverlay_Draw(text, p, stale) {
	var textObj = Engine.GetGUIObjectByName("coachWinProbText");
	var bar = Engine.GetGUIObjectByName("coachWinProbBar");
	var bg = Engine.GetGUIObjectByName("coachWinProbBarBg");
	if (!textObj || !bar || !bg) return;
	textObj.caption = text;
	textObj.textcolor = stale ? "160 160 160" : "white";
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
	// green when ahead, red when behind, grey when stale
	bar.sprite = stale ? "color: 140 140 140 200"
		: p >= 0.5 ? "color: 90 200 90 230" : "color: 220 80 80 230";
}

function CoachOverlay_ReadBack() {
	try {
		var d = Engine.ReadJSONFile(g_CoachOverlay_InPath);
		if (!d || typeof d.p !== "number") {
			CoachOverlay_Draw("coach: no data", null, true);
			return;
		}
		var stale = !d.written_at_ms || (Date.now() - d.written_at_ms) > g_CoachOverlay_StaleMs;
		CoachOverlay_Draw("Win " + Math.round(d.p * 100) + "%" + (stale ? " (stale)" : ""), d.p, stale);
	} catch (e) {
		warn("[coach-overlay] readback failed: " + e);
	}
}

function CoachOverlay_Loop() {
	if (!CoachOverlay_IsReady()) {
		setTimeout(CoachOverlay_Loop, 1000);
		return;
	}
	CoachOverlay_Publish();
	CoachOverlay_ReadBack();
	setTimeout(CoachOverlay_Loop, g_CoachOverlay_IntervalMs);
}

function CoachOverlay_Boot() {
	if (g_CoachOverlay_Started) return;
	g_CoachOverlay_Started = true;
	CoachOverlay_InitInbox();
	// Small initial delay so session.js has finished its first init pass.
	setTimeout(CoachOverlay_Loop, 500);
}

// The session GUI loads every JS file in gui/session/. Kicking off the loop
// at module load is safe: setTimeout defers work until the event loop runs.
CoachOverlay_Boot();
