// v1 : lecture de la Live Client Data API de League of Legends.
//
// Pendant une partie, le jeu expose un petit serveur HTTPS local sur
// https://127.0.0.1:2999 . On l'interroge une fois par seconde, on affiche
// l'état de la partie, les nouveaux événements, et deux minuteurs simples
// (dragon, Baron). Lecture seule : rien n'est envoyé au jeu.
//
// Lancer :  node v1.mjs   (avec une partie ou le Practice Tool ouvert)

import https from "node:https";

const URL = "https://127.0.0.1:2999/liveclientdata/allgamedata";
const INTERVAL_MS = 1000;

// Le certificat du serveur local est signé par Riot, pas par une autorité
// connue de Node : on désactive la vérification pour cette seule connexion.
const agent = new https.Agent({ rejectUnauthorized: false });

function fetchGameData() {
  return new Promise((resolve, reject) => {
    https
      .get(URL, { agent }, (res) => {
        let body = "";
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(body));
          } catch (err) {
            reject(err);
          }
        });
      })
      .on("error", reject);
  });
}

function mmss(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

// --- Minuteurs d'objectifs (la couche "décision", réduite au minimum) ---
// Dragon : premier à 5:00, puis 5:00 après chaque mort.
// Baron  : premier à 20:00, puis 6:00 après chaque mort.
function nextSpawns(events) {
  let dragon = 5 * 60;
  let baron = 20 * 60;
  for (const e of events) {
    if (e.EventName === "DragonKill") dragon = e.EventTime + 5 * 60;
    if (e.EventName === "BaronKill") baron = e.EventTime + 6 * 60;
  }
  return { dragon, baron };
}

function describe(e) {
  const t = mmss(e.EventTime);
  switch (e.EventName) {
    case "GameStart":
      return `${t}  La partie commence`;
    case "MinionsSpawning":
      return `${t}  Les sbires apparaissent`;
    case "FirstBlood":
      return `${t}  Premier sang : ${e.Recipient}`;
    case "ChampionKill":
      return `${t}  ${e.KillerName} tue ${e.VictimName}`;
    case "DragonKill":
      return `${t}  Dragon (${e.DragonType}) tué par ${e.KillerName}`;
    case "BaronKill":
      return `${t}  Baron tué par ${e.KillerName}`;
    case "HeraldKill":
      return `${t}  Héraut tué par ${e.KillerName}`;
    case "TurretKilled":
      return `${t}  Tour détruite par ${e.KillerName}`;
    case "InhibKilled":
      return `${t}  Inhibiteur détruit par ${e.KillerName}`;
    case "Ace":
      return `${t}  Ace pour l'équipe ${e.AcingTeam}`;
    default:
      return `${t}  ${e.EventName}`;
  }
}

let lastEventId = -1;
let waiting = false;

function render(data) {
  const time = data.gameData.gameTime;
  const me = data.activePlayer;
  const events = data.events.Events;
  const { dragon, baron } = nextSpawns(events);

  console.clear();
  console.log(`=== ${data.gameData.gameMode}  ${mmss(time)} ===\n`);
  console.log(
    `Moi : ${me.riotIdGameName ?? me.summonerName}  niveau ${me.level}  or ${Math.floor(me.currentGold)}\n`,
  );

  console.log("Équipe  Champion          Niv  K/D/A     CS");
  for (const p of data.allPlayers) {
    const s = p.scores;
    console.log(
      `${p.team.padEnd(7)} ${p.championName.padEnd(17)} ${String(p.level).padStart(3)}  ` +
        `${`${s.kills}/${s.deaths}/${s.assists}`.padEnd(9)} ${s.creepScore}`,
    );
  }

  console.log("\nProchains objectifs :");
  console.log(`  Dragon dans ${mmss(dragon - time)}`);
  console.log(`  Baron  dans ${mmss(baron - time)}`);

  console.log("\nDerniers événements :");
  for (const e of events.slice(-8)) console.log("  " + describe(e));

  // Les événements nouveaux depuis le dernier passage sont signalés à part.
  const fresh = events.filter((e) => e.EventID > lastEventId);
  if (lastEventId >= 0 && fresh.length) {
    console.log("\n>>> NOUVEAU :");
    for (const e of fresh) console.log("  " + describe(e));
  }
  if (events.length) lastEventId = events.at(-1).EventID;
}

async function tick() {
  try {
    const data = await fetchGameData();
    waiting = false;
    render(data);
  } catch (err) {
    if (!waiting) {
      console.clear();
      console.log("En attente d'une partie (le serveur local répond sur le port 2999 dès qu'une partie est chargée)...");
      console.log(`Détail : ${err.code ?? err.message}`);
      waiting = true;
      lastEventId = -1;
    }
  }
}

tick();
setInterval(tick, INTERVAL_MS);
