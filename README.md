# Rust Képfeldolgozó, Párhuzamosító és GPU Teljesítménytesztelő Rendszer

Ez a szoftver a szakdolgozatom gyakorlati implementációját képezi. A projekt célja különböző képfeldolgozási algoritmusok végrehajtási idejének precíz mérése, valamint a szekvenciális (egyszálú) CPU, a többmagos (párhuzamos) CPU, valamint a hardveresen gyorsított GPGPU (grafikus kártya) architektúrák teljesítményének összehasonlítása.

## Alkalmazott Technológiák és Architektúrák

- **CPU Párhuzamosítás:** A Rust **Rayon** könyvtár segítségével, adatpárhuzamos (data-parallel) módon, munkalopásos (work-stealing) ütemezéssel osztja szét a pixelmátrixokat a logikai processzormagok között.
- **Hardveres GPU Gyorsítás:** A modern, alacsony szintű **wgpu** API segítségével közvetlen hozzáférés a grafikus hardverhez, a compute shaderek pedig natív **WGSL** (WebGPU Shading Language) nyelven futnak a videókártya magjain.
- **CLI Interfész:** Robusztus parancssori argumentum-feldolgozás a **Clap** könyvtár használatával.

## Támogatott Algoritmusok (Szűrők)

1. **Negatív effekt (CPU & GPU/WGSL):** Színcsatornák invertálása bit szintű operációkkal ($255 - x$).
2. **Szürkeárnyalatosítás (CPU & GPU/WGSL):** Luminancia-alapú szürkeárnyalatos konverzió a BT.601 szabvány súlyozásával ($0.299 \times R + 0.587 \times G + 0.114 \times B$).
3. **Fényerő növelése (CPU):** Fényerő eltolás túlcsordulás-biztos (`saturating_add`) aritmetikával.
4. **Sobel élkeresés (CPU & GPU/WGSL):** Gradiens-alapú élkeresés vízszintes ($G_x$) és függőleges ($G_y$) Sobel-operátorok konvolúciójával, és a magnitúdó kiszámításával.
5. **Box Blur (CPU):** Képsimítás és zajcsökkentés $3 \times 3$-as környezeti átlagolással.

## Mérés és Adatgyűjtés

A program futás végén nanoszekundumos pontossággal kiértékeli a végrehajtási időket, és kiszámítja a CPU-s gyorsulási faktort (Speedup).
Az automatizált kiértékelés érdekében a rendszer a futási adatokat strukturált **`meresek.csv`** fájlba exportálja a projekt gyökerébe, ami azonnal importálható Excelbe vagy egyéb adatvizualizációs szoftverekbe diagramok generálásához.

## Használati Útmutató

### Előfeltételek
- Működő Rust eszközkészlet (`cargo`, `rustc` v1.70+).
- Vulkan, Metal vagy DirectX 12 támogatással rendelkező grafikus vezérlő és friss driverek (a `wgpu` futtatásához).

### Futtatás parancssorból
A mérések pontossága érdekében elengedhetetlen a fordítói optimalizáció, a programot mindig a `--release` kapcsolóval kell futtatni!

A megújult CLI interfész lehetővé teszi a bemeneti kép és a mentési útvonal rugalmas megadását paramétereken keresztül:

```bash
cargo run --release -- --image bemenet.jpg --filter all --mode cpu
