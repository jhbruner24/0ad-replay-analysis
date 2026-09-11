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

function CoachOverlay_Loop() {
	if (!CoachOverlay_IsReady()) {
		setTimeout(CoachOverlay_Loop, 1000);
		return;
	}
	CoachOverlay_Publish();
	setTimeout(CoachOverlay_Loop, g_CoachOverlay_IntervalMs);
}

function CoachOverlay_Boot() {
	if (g_CoachOverlay_Started) return;
	g_CoachOverlay_Started = true;
	// Small initial delay so session.js has finished its first init pass.
	setTimeout(CoachOverlay_Loop, 500);
}

// The session GUI loads every JS file in gui/session/. Kicking off the loop
// at module load is safe: setTimeout defers work until the event loop runs.
CoachOverlay_Boot();
