# Overview

## We are working on a uni project. These are the general requirements:

Předpokládá se načtení parametrů a dalších dat ze souboru (dává-li to smysl, umožnit jeho vygenerování/editaci) a/nebo jejich interaktivní zadaní, následně spuštění (dává-li to smysl, i krokování) algorimu, ideálně s půběžným vypisováním podstatných informací o průběhu, s možnost interaktivního ukončnení, následně proběhne zobrazení výsledků. Tento základní princip lze v konkrétních případech vhodně vylepšit a upravit.

## This is the specific topic that we have chosen for our project:

You are working on a modern Hopfield network–based demo for **robot localization** on a top-down map. The idea is to represent the environment as a visual map, with the robot moving inside it. The robot “sees” a small local observation, such as a line or patch of pixels from the map (e.g., a strip of colors corresponding to what a camera would capture). These observations are stored in memory during a setup phase, effectively building a database of known positions and their associated observations. Then, given a new observation, the modern Hopfield network retrieves the closest matching stored pattern, allowing the robot to determine its approximate position on the map in real time.

## General instructions

-   More context can be found in the `docs` folder. The whole specification is/will be stored there.
