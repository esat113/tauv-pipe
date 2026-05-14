# Pipe Tracker v4 Mimarisi

Bu dokuman `pipe_tracker_gui4.py`, `pipe_tracker_dds.py`, `pipe_tracker_aruco.py` ve `pipe_algorithm.py` icindeki yeni boru takip sisteminin nasil calistigini anlatir. Hedef, SAM3'ten gelen boru maskesini dogrudan bir yaw PID'ine vermek yerine once geometrik takip olcumlerine cevirmek, sonra bu olcumleri dogru RC eksenlerine dagitmaktir.

## Kisa Ozet

V4 sistem su akisi izler:

```text
bottom camera frame
        +
SAM3 segmentation mask
        |
        v
GUI overlay / timestamp sync
        |
        v
PipeGeometryProcessor
    - mask temizleme
    - en buyuk component secimi
    - slice centroid'leri
    - centerline fit
    - lateral/heading/lookahead hatalari
        |
        v
PipeVisualServoController
    - SEARCH / ACQUIRE / TRACK / LOST state
    - EMA filtreleme
    - P tipi ayrik visual-servo kontrol
    - slew-rate limit
    - forward hiz planlama
        |
        v
DDS StreamCommand: motor_rc
        |
        v
ArduSub RC override
```

En onemli tasarim degisikligi sudur:

```text
Eski yaklasim:
  maskenin merkez hatasi -> yaw

V4 yaklasimi:
  boru merkez hatasi       -> lateral
  boru acisi / lookahead   -> yaw
  takip guveni / hata buyuklugu -> forward hiz
```

Bu ayrim alt kamera ile boru takibinde kritik. Alt kamerada boru goruntude saga kaydiysa aracin once saga/sola kaymasi gerekir; sadece yaw verirse arac boruya dogru doner, boruya paralel kalamaz ve salinim artar.

## Dosyalar

- `pipe_tracker_gui4.py`: SAM3 prompt, overlay, tuning UI ve takip dongusu.
- `pipe_tracker_dds.py`: V4'e ait DDS kamera/mask okuma tipleri ve `motor_rc` publish katmani.
- `pipe_tracker_aruco.py`: `DICT_ARUCO_ORIGINAL` marker tespiti, overlay cizimi ve GUI okuma sirasi.
- `pipe_algorithm.py`: Maske geometrisi ve takip kontrolcusu.
- `tests/test_pipe_visual_servo.py`: Sentetik maske testleri. Kodun isaretlerini ve temel davranisini dogrular.
- `tests/test_pipe_aruco.py`: Sentetik ArUco marker testleri ve okuma sirasi dogrulamasi.

V4 GUI, eski `pipe_tracker_gui.py`, `pipe_tracker_gui2.py` veya `pipe_tracker_gui3.py` dosyalarindan kod import etmez. Bu eski dosyalar referans/onceki surum olarak kalabilir, ancak guncel V4 calisma yolu `pipe_tracker_gui4.py` + `pipe_tracker_dds.py` + `pipe_tracker_aruco.py` + `pipe_algorithm.py` uzerindedir.

## Temel Kavramlar

**Segmentation mask**

SAM3'in urettigi ikili goruntu. Boru pikselleri `1` veya `255`, arka plan `0` olabilir. V4 iki formati da kabul eder. GUI maskeyi kamera frame boyutuna nearest-neighbor ile resize eder.

**Component**

Maskede birbirine bagli piksellerin olusturdugu parca. Algoritma en buyuk component'i boru kabul eder. Bu, kucuk gurultu lekelerini yok saymak icin kullanilir.

**Centroid**

Bir mask parcasinin agirlik merkezi. V4 tum maskenin tek centroid'i ile yetinmez; maskeyi yatay dilimlere ayirip her dilimde centroid bulur.

**Centerline**

Boru maskesinin goruntudeki orta cizgisi. V4, dilim centroid'lerine `x = slope * y + intercept` dogrusu fit eder. Bu cizgi borunun goruntudeki yonunu ve merkezden kaymasini verir.

**Lateral error**

Aracin alt kamera goruntusunde boru merkezinin yatay merkeze gore kaymasidir.

```text
lateral_error > 0  -> boru goruntude sagda
lateral_error < 0  -> boru goruntude solda
lateral_error = 0  -> boru merkezde
```

Bu hata `[-1, +1]` araligina normalize edilir. `+1`, goruntunun yaklasik sag kenari demektir.

**Lookahead error**

Boru cizgisinin goruntunun daha yukari/ileri tarafindaki noktasinin merkezden sapmasidir. Bu, "biraz sonra boru nereye gidiyor?" bilgisidir. Tek basina lateral merkezleme degil, ilerideki yon degisimini yakalamak icin kullanilir.

**Heading error**

Boru centerline'inin goruntudeki acisidir. Borunun ileri ucu saga dogru gidiyorsa pozitif, sola gidiyorsa negatif kabul edilir.

**Confidence**

Maskeden cikan olcume guven skorudur. Dilim sayisi, alan ve elongation beraber degerlendirilir. `min_confidence` altinda kalan geometri takip icin kullanilmaz.

**PWM / RC override**

ArduSub motor kanallarina `motor_rc` komutu ile PWM degeri gonderilir. Nötr deger `1500` kabul edilir. V4 su kanallari kullanir:

```python
{
    "pitch": 1500,
    "roll": 1500,
    "throttle": 1500,
    "yaw": 1500 + yaw_offset,
    "forward": 1500 + fwd_offset,
    "lateral": 1500 + lateral_offset,
}
```

`pitch`, `roll`, `throttle` sabit nötr tutulur. Derinlik kontrolu veya attitude kontrolu bu GUI'nin isi degildir.

## GUI Tarafi

`PipeTrackerServoWindow` su isleri yapar:

1. `camera/bottom/frame` topic'inden kamera frame alir.
2. `sam3/bottom/segmentation_mask` topic'inden SAM3 maskesi alir.
3. Mask timestamp'i ile kamera frame timestamp'ini eslestirmeye calisir.
4. Eslesme yoksa son kamera frame'i ile fallback yapar.
5. Kamera frame de yoksa maskeyi tek basina gosterir.
6. Maskeyi overlay eder ve geometri cizimlerini ekler.
7. Kamera frame varsa ayni yeni SAM3 mask frame'inde ArUco marker tespiti yapar.
8. Tespit edilen ArUco ID'lerini kamera goruntusune ve ayri GUI satirina yazar.
9. Takip aciksa `PipeVisualServoController.compute()` ile RC komutu uretir.
10. `embedded/control/stream_command` topic'ine `motor_rc` yayinlar.

ArUco tespiti sadece gorsellestirme ve operator bilgisi icindir. Marker ID'leri veya marker konumu boru takip kontrolcusune verilmez; yaw, lateral ve forward komutlari sadece maske geometrisinden uretilir.

GUI'nin ust bilgi satirinda su debug bilgileri gorulur:

```text
mask=270x203 nz=12345 max=255 frame=matched geom=ok
lat=+0.12 look=+0.20 ang=+8.5 conf=0.91 age=0.03s mask_fps=9.8
yaw=1526 fwd=1580 lat=1511
```

Alanlar:

- `mask=WxH`: DDS'ten gelen maskenin boyutu.
- `nz`: non-zero piksel sayisi. `0` ise SAM3 boru bulmuyor veya prompt/threshold yanlis.
- `max`: maskenin maksimum piksel degeri. `1` veya `255` olabilir.
- `frame`: `matched`, `fallback` veya `none(mask-only)`.
- `geom`: geometri cikarma sonucu. `ok`, `small_area`, `too_few_slices`, `no_component` gibi.
- `age`: son kullanilan geometrinin yasi. Stale olursa kontrol SEARCH/LOST tarafina gider.

## Maske Geometrisi

`PipeGeometryProcessor.process(mask)` motor komutu uretmez. Sadece maskeyi takip edilebilir bir olcume cevirir.

### 1. Binary mask normalize

Mask `0/1` gelirse `0/255`'e cevrilir. Mask `0/255` gelirse `128` threshold ile binary hale getirilir.

### 2. Morfolojik temizlik

`cv2.morphologyEx` ile once close, sonra open uygulanir.

- Close: boru maskesindeki kucuk delikleri kapatir.
- Open: tekil gurultu lekelerini temizler.

Kernel boyutu `morph_kernel_ratio` ile mask boyutuna gore hesaplanir.

### 3. En buyuk component secimi

`connectedComponentsWithStats` ile bagli parcaciklar bulunur. En buyuk alanli component boru kabul edilir. Bu component cok kucukse `small_area` doner.

### 4. Slice centroid'leri

Mask yatay dilimlere ayrilir. Her dilimde yeterli alan varsa centroid hesaplanir.

```text
top
 ┌───────────────┐
 │      x        │  slice 0 centroid
 ├───────────────┤
 │       x       │  slice 1 centroid
 ├───────────────┤
 │        x      │  slice 2 centroid
 └───────────────┘
bottom
```

Bu yontem tek centroid'e gore daha kararlidir; borunun goruntudeki acisini ve egimini verir.

### 5. Centerline fit

Centroid noktalarina agirlikli dogru fit edilir:

```text
x = slope * y + intercept
```

Agirlik olarak dilimdeki maske alani kullanilir. Daha dolu dilimler fit uzerinde daha etkili olur.

### 6. Takip olcumleri

Fit edilen dogru uzerinden uc nokta hesaplanir:

- `x_center`: `center_y_ratio` seviyesindeki boru merkezi.
- `x_look`: `lookahead_y_ratio` seviyesindeki ileri nokta.
- `x_top`, `x_bottom`: boru acisi icin ust ve alt uc.

Sonra:

```text
lateral_error = (x_center - image_center_x) / half_width
lookahead_error = (x_look - image_center_x) / half_width
heading_error_deg = atan2(x_top - x_bottom, y_bottom - y_top)
```

## Kontrol Mimarisi

`PipeVisualServoController` klasik anlamda tam bir PID degildir. Bilerek daha basit ve daha kararlı bir **ayrik P kontrol + EMA filtre + slew-rate limit + state machine** kullaniyoruz.

Neden tam PID degil?

- SAM3 yaklasik 10 FPS calisiyor.
- Segmentasyon maskesi frame-to-frame jitter uretebilir.
- D terimi jitter'i buyutur.
- I terimi sabit bias'i duzeltebilir ama su an goruntu hatasinda windup ve gecikme riski fazla.

Bu yuzden V4'te ana takip su sekilde:

```text
filtered_lat   = EMA(lateral_error)
filtered_look  = EMA(lookahead_error)
filtered_angle = EMA(heading_error_deg)

lateral_offset = Kp_lateral * filtered_lat

look_delta = filtered_look - filtered_lat
yaw_offset = Kp_yaw_angle * filtered_angle
           + Kp_yaw_lookahead * look_delta
```

### Neden `look_delta = look - lat`?

Eger boru komple sagdaysa:

```text
lat  = +0.40
look = +0.40
```

Bu durumda boru paralel ama sagda demektir. Yaw vermek yerine lateral kaymak daha dogrudur. `look_delta = 0` olur ve yaw komutu kucuk kalir.

Eger boru ileri tarafta saga kiriliyorsa:

```text
lat  = +0.05
look = +0.30
```

Bu durumda boru onumuzde saga gidiyor demektir. `look_delta > 0` olur ve yaw komutu uretilir.

Bu ayrim eski sistemdeki en buyuk problemi cozer: yanal merkezleme ve heading hizalama birbirinden ayrilir.

## State Machine

Kontrolcu dort state kullanir.

### SEARCH

Gecerli boru yoksa arama durumudur. Forward verilmez, yavas yaw taramasi yapilir.

```text
forward = 1500
yaw = 1500 +/- search_yaw_pwm
lateral = 1500
```

`search_switch_s` suresinde bir yaw yonu degisir.

### ACQUIRE

Boru yeni bulunduysa hemen tam hiz takip yapilmaz. `acquire_frames` kadar iyi frame beklenir. Forward daha dusuktur.

Bu durum, tek frame'lik yanlis SAM3 maskesinin araci sert hareket ettirmesini engeller.

### TRACK

Normal takip durumudur. Geometri guvenliyse lateral, yaw ve forward komutlari uretilir.

### LOST

Boru kisa sure kaybolursa son yaw/lateral komutlari yavasca sondurulur. `lost_hold_s` suresi asilirsa SEARCH'e doner.

## RC Komutlari

V4 sadece `motor_rc` stream command yayinlar:

```python
StreamCommand(
    command_type="motor_rc",
    command_data=json.dumps({
        "pitch": 1500,
        "roll": 1500,
        "throttle": 1500,
        "yaw": 1500 + yaw_offset,
        "forward": 1500 + fwd_offset,
        "lateral": 1500 + lateral_offset,
    }),
    client_id="pipe_tracker_gui4",
)
```

Kanal anlamlari:

| Kanal | V4'teki rol |
| --- | --- |
| `pitch` | Nötr. Kullanilmiyor. |
| `roll` | Nötr. Kullanilmiyor. |
| `throttle` | Nötr. Derinlik kontrolune dokunmuyoruz. |
| `yaw` | Boru acisi ve lookahead farkina gore donus. |
| `forward` | Boru boyunca ilerleme hizi. |
| `lateral` | Boruyu goruntu merkezine almak icin saga/sola kayma. |

Varsayilan offset limitleri:

- `max_yaw_pwm = 120`
- `max_lateral_pwm = 120`
- `base_forward_pwm = 105`
- `acquire_forward_pwm = 35`

Yani normal takipte komutlar kabaca su araliklarda kalir:

```text
yaw     : 1380 .. 1620
lateral : 1380 .. 1620
forward : 1500 .. 1605 civari
```

## Forward Hiz Planlama

Forward sabit degildir. Hata buyudukce ve boru acisi arttikca azalir.

Kod mantigi:

```text
angle_penalty = min(1, abs(angle) / 45)
error_penalty = min(1, abs(lat))

speed_scale = 1 - 0.55 * error_penalty - 0.35 * angle_penalty
speed_scale = clamp(speed_scale, min_forward_scale, 1)
```

Sonra:

```text
ACQUIRE:
  fwd = min(acquire_forward_pwm, base_forward_pwm * speed_scale)

TRACK:
  fwd = base_forward_pwm * speed_scale * max(0.35, confidence)
```

Bu su anlama gelir:

- Boru merkezde ve paralelse ileri hiz artar.
- Boru cok yandaysa once lateral/yaw duzeltme yapilir, forward azalir.
- Confidence dusukse aracin agresif ileri gitmesi engellenir.

## EMA Filtresi

EMA, Exponential Moving Average demektir. Yeni olcum ile onceki filtreli degerin karisimidir.

```text
filtered = alpha * new_value + (1 - alpha) * previous_filtered
```

`ema_alpha = 0.45` varsayilanidir.

- Alpha buyukse sistem daha hizli tepki verir ama jitter artabilir.
- Alpha kucukse daha yumusak gider ama gecikir.

Bu filtre SAM3 maskesinin frame-to-frame ziplamasini azaltir.

## Slew Rate Limit

Slew-rate limit, komutun bir anda cok degismesini engeller.

```text
max_delta = slew_pwm_per_s * dt
```

Ornegin `slew_pwm_per_s = 500` ve `dt = 0.1s` ise bir frame'de en fazla `50 PWM` degisim olur.

Bu, motorlara ani sert komut gitmesini ve su altinda osilasyonu azaltir.

## Tuning Parametreleri

GUI uzerinden canli degistirilen ana parametreler:

| Parametre | Anlam |
| --- | --- |
| `kp_lateral` | `lateral_error` -> lateral PWM kazanci. |
| `kp_yaw_angle` | Boru acisi -> yaw PWM kazanci. |
| `kp_yaw_lookahead` | Ileri noktanin merkezden sapma farki -> yaw PWM kazanci. |
| `base_forward_pwm` | TRACK durumundaki temel ileri hiz offset'i. |
| `acquire_forward_pwm` | ACQUIRE durumundaki dusuk ileri hiz offset'i. |
| `max_lateral_pwm` | Lateral offset saturasyonu. |
| `max_yaw_pwm` | Yaw offset saturasyonu. |
| `min_confidence` | Bunun altindaki geometri takipte kullanilmaz. |
| `ema_alpha` | Olcum filtresi. |
| `yaw_sign` | Yaw yonu tersse `-1` yapilir. |
| `lateral_sign` | Lateral yonu tersse `-1` yapilir. |
| `mask_stale_s` | Son mask bu sureden eskiyse kontrol stale sayar. |
| `lookahead_y_ratio` | Lookahead noktasinin goruntudeki y seviyesi. |
| `center_y_ratio` | Lateral merkezleme noktasinin y seviyesi. |
| `min_area_ratio` | Boru kabul edilecek minimum mask alan orani. |

Pratik tuning sirasi:

1. Arac su icinde guvenli ve dusuk hizda olsun.
2. `forward` dusuk baslasin.
3. Boru sagdaysa `lateral` dogru yone gidiyor mu bak. Tersse `lateral_sign = -1`.
4. Boru ileri tarafta saga kiriliyorsa yaw dogru yone gidiyor mu bak. Tersse `yaw_sign = -1`.
5. Lateral gec kaliyor ama dogru gidiyorsa `kp_lateral` yavas yavas artirilir.
6. Virajlara gec kaliyor ama salinim yoksa `kp_yaw_angle` veya `kp_yaw_lookahead` artirilir.
7. Titreme varsa once `ema_alpha` dusurulur veya `max_yaw_pwm/max_lateral_pwm` azaltılır.
8. Takip stabilse `base_forward_pwm` kademeli artirilir.

## Neden PID Yerine Bu Mekanizma?

Burada hala "P kontrol" var; fakat klasik PID degil.

Klasik PID:

```text
output = Kp * error + Ki * integral(error) + Kd * derivative(error)
```

V4:

```text
lateral_output = Kp_lateral * filtered_lateral_error
yaw_output = Kp_angle * filtered_heading_error
           + Kp_lookahead * (filtered_lookahead_error - filtered_lateral_error)
```

Integral yok:

- SAM3 gecikmeli ve 10 FPS civari calisiyor.
- Integral gecikmeli sistemlerde windup yapabilir.
- Arac zaten su icinde atalete sahip; once basit ve tahmin edilebilir kontrol tercih edildi.

Derivative yok:

- Segmentasyon jitter'i derivative terimde buyur.
- D terimi motor komutunu ziplatabilir.

Bunun yerine:

- EMA ile olcum yumusatildi.
- Slew-rate ile cikis yumusatildi.
- State machine ile boru yokken guvenli davranis verildi.

## Testler Neyi Garanti Ediyor?

`tests/test_pipe_visual_servo.py` sentetik maskelerle su davranislari kontrol eder:

- Ortali dik boruda forward verilir, yaw/lateral yaklasik nötr kalir.
- Saga kaymis ama paralel boruda yaw yerine lateral komut baskin olur.
- Capraz boruda yaw komutu uretilir.
- Maske yoksa SEARCH state'e gecilir ve forward verilmez.
- `0/1` formatindaki maskeler de kabul edilir.

Bu testler hidrodinamik davranisi garanti etmez; ama algoritmanin temel isaretleri ve eksen dagilimi dogru mu bunu yakalar.

## Bilinen Sinirlar

- Kamera kalibrasyonu ve piksel-hata/metre donusumu yok. Her sey goruntu uzayinda normalize edilmis hata ile calisiyor.
- Boru keskin viraj veya T-junction yaparsa tek dogru fit yeterli olmayabilir.
- SAM3 yanlis nesneyi pipe diye maskelerse en buyuk component stratejisi onu takip eder.
- Forward hiz PWM offset olarak veriliyor; gercek m/s kapali devre kontrol yok.
- Lateral ve yaw eksenleri arac/motor mapping'ine bagli. Bu yuzden `yaw_sign` ve `lateral_sign` tuning alanlari var.

## Debug Ipuclari

GUI bilgi satirinda:

- `mask=YOK`: DDS mask gelmiyor. SAM3 topic, DDS config veya container kontrol edilmeli.
- `nz=0`: mask geliyor ama SAM3 boru bulmuyor. Prompt/threshold/postprocess kontrol edilmeli.
- `frame=none(mask-only)`: mask geliyor ama kamera frame gelmiyor.
- `geom=small_area`: mask var ama alan threshold altinda.
- `geom=too_few_slices`: mask var ama centerline fit icin yeterli dilim yok.
- `conf` dusuk: takip SEARCH/LOST davranisina gecebilir.

Kontrol komutunda:

- Boru sagda ama `lateral < 1500` ise `lateral_sign = -1` denenmeli.
- Boru saga donuyor ama `yaw < 1500` ise `yaw_sign = -1` denenmeli.
- Titreme varsa `base_forward_pwm` azalt, `ema_alpha` dusur, `max_yaw_pwm/max_lateral_pwm` dusur.
