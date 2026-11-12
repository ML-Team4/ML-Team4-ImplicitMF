# ImplicitMF 정성적 추천 결과 리포트

## User `0` (학습 기간 총 799곡 청취)

### 1. 학습 기간 주요 청취 이력 (Top-5)

| Rank | Track Title | Artist | Play Count | Train Listeners |
| :---: | :--- | :--- | :---: | :---: |
| 1 | Amicalement Vôtre | Tour Maubourg | 7회 | 1명 |
| 2 | Floating On Silence | Tour-Maubourg | 7회 | 4명 |
| 3 | Allégresse | Tour Maubourg | 7회 | 1명 |
| 4 | Kindness | Retromigration & Nephews | 7회 | 2명 |
| 5 | You Like Jazz? | Diamond | 7회 | 1명 |

### 2. Top-10 추천 목록 — Baseline (IPS 미적용) (평균 청취자 수: **128.1명**)

| Rank | Item ID | Track Title | Artist | Train Listeners |
| :---: | :---: | :--- | :--- | :---: |
| 1 | `1175443` | Everybody Wants to Rule the World | Tears for Fears | 240명 |
| 2 | `807109` | Billie Jean | Michael Jackson | 265명 |
| 3 | `1355863` | Maps | Yeah Yeah Yeahs | 56명 |
| 4 | `1248302` | All That We Perceive | Thievery Corporation | 25명 |
| 5 | `807094` | Beat It | Michael Jackson | 220명 |
| 6 | `1248844` | Semi‐Charmed Life | Third Eye Blind | 91명 |
| 7 | `45649` | Kelly Watch the Stars | Air | 29명 |
| 8 | `1105316` | Before I Forget | Slipknot | 88명 |
| 9 | `254758` | Don’t Panic | Coldplay | 62명 |
| 10 | `1383767` | Take On Me | a‐ha | 205명 |

### 2. Top-10 추천 목록 — IPS 적용 ($\gamma=0.20$) (평균 청취자 수: **70.7명**)

| Rank | Item ID | Track Title | Artist | Train Listeners |
| :---: | :---: | :--- | :--- | :---: |
| 1 | `1175443` | Everybody Wants to Rule the World | Tears for Fears | 240명 |
| 2 | `760483` | Like a Virgin | Madonna | 34명 |
| 3 | `62312` | I Need a Dollar | Aloe Blacc | 23명 |
| 4 | `254758` | Don’t Panic | Coldplay | 62명 |
| 5 | `1248302` | All That We Perceive | Thievery Corporation | 25명 |
| 6 | `19074` | Jazz (We’ve Got) | A Tribe Called Quest | 36명 |
| 7 | `1248844` | Semi‐Charmed Life | Third Eye Blind | 91명 |
| 8 | `1225606` | Your Love | The Outfield | 88명 |
| 9 | `1385131` | Sun Has Set | beabadoobee | 73명 |
| 10 | `1248354` | Lebanese Blonde | Thievery Corporation | 35명 |

---

## User `10` (학습 기간 총 6,950곡 청취)

### 1. 학습 기간 주요 청취 이력 (Top-5)

| Rank | Track Title | Artist | Play Count | Train Listeners |
| :---: | :--- | :--- | :---: | :---: |
| 1 | U I Luv | Coco Bryce | 34회 | 2명 |
| 2 | Dread | Breakage | 25회 | 1명 |
| 3 | Loud Places (Special Request VIP) | Jamie xx feat. Romy | 22회 | 2명 |
| 4 | Pull Up (feat. Pino D’Angiò & Scrufizzer) | Young Franco | 17회 | 1명 |
| 5 | Techno Syndrome (Mortal Kombat) | The Immortals | 15회 | 14명 |

### 2. Top-10 추천 목록 — Baseline (IPS 미적용) (평균 청취자 수: **34.0명**)

| Rank | Item ID | Track Title | Artist | Train Listeners |
| :---: | :---: | :--- | :--- | :---: |
| 1 | `433482` | Two Thousand and Seventeen | Four Tet | 19명 |
| 2 | `824632` | Porcelain | Moby | 84명 |
| 3 | `175903` | Always on Your Side | Bonobo & Joy Crookes | 21명 |
| 4 | `1228339` | Out of Space | The Prodigy | 42명 |
| 5 | `824726` | Why Does My Heart Feel So Bad? | Moby | 44명 |
| 6 | `213573` | Home | Caribou | 14명 |
| 7 | `1285036` | Breathe | Télépopmusik | 26명 |
| 8 | `1291264` | Born Slippy .NUXX | Underworld | 43명 |
| 9 | `213598` | Odessa | Caribou | 22명 |
| 10 | `979532` | Time Is the Enemy | Quantic | 25명 |

### 2. Top-10 추천 목록 — IPS 적용 ($\gamma=0.20$) (평균 청취자 수: **48.7명**)

| Rank | Item ID | Track Title | Artist | Train Listeners |
| :---: | :---: | :--- | :--- | :---: |
| 1 | `824632` | Porcelain | Moby | 84명 |
| 2 | `175903` | Always on Your Side | Bonobo & Joy Crookes | 21명 |
| 3 | `433482` | Two Thousand and Seventeen | Four Tet | 19명 |
| 4 | `213573` | Home | Caribou | 14명 |
| 5 | `1285036` | Breathe | Télépopmusik | 26명 |
| 6 | `824726` | Why Does My Heart Feel So Bad? | Moby | 44명 |
| 7 | `980105` | Bohemian Rhapsody | Queen | 196명 |
| 8 | `1291264` | Born Slippy .NUXX | Underworld | 43명 |
| 9 | `427157` | Birth4000 | Floating Points | 11명 |
| 10 | `170093` | Aquarius | Boards of Canada | 29명 |

---
