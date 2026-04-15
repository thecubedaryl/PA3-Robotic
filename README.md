# PA3-Robotic (Duckietown Assignment 3)

## Proje Özeti

Bu proje, Duckietown Assignment 3'ün bir Duckiebot üzerinde gerçekleştirilmesini amaçlamaktadır.

Projede kullanılan temel bileşenler:

* `N0` noktasından `N15` noktasına en kısa yolu bulmak için **A*** algoritması
* Hedef düğümü algılamak için **AprilTag (ARTag) tabanlı konumlama**
* Robotu planlanan yol boyunca ilerletmek için **ROS tabanlı hareket kontrolü**

Navigator düğümü, yolu önce hesaplar ve ardından gerçek zamanlı AprilTag algılamaları ile robotu düğümden düğüme yönlendirerek hedefe ulaştırır.

---

## Dosya Yapısı

```text
PA3_ROBOTIC/
├── README.md
└── assignment3/
    ├── Dockerfile
    ├── dt-project.yaml
    ├── configurations.yaml
    ├── dependencies-apt.txt
    ├── dependencies-py3.txt
    ├── dependencies-py3.dt.txt
    ├── launchers/
    │   └── default.sh
    └── packages/
        └── assignment3/
            ├── CMakeLists.txt
            ├── package.xml
            ├── launch/
            │   └── assignment3.launch
            └── src/
                ├── astar.py
                └── navigator_node.py
```

---

## Ana Dosyalar

* **astar.py**
  A* algoritmasını uygular ve hesaplanan yolu ile toplam maliyeti ekrana yazdırır.

* **navigator_node.py**
  Ana ROS düğümüdür. Yolu yükler, AprilTag verilerini dinler ve hareket komutlarını yayınlar.

* **assignment3.launch**
  Navigator düğümünü başlatır.

* **default.sh**
  Duckietown çalıştırma scriptidir (`dts devel run` ile kullanılır).

---

## Robot Adı Nasıl Değiştirilir

Robot adınız `autobot01` değilse aşağıdaki yerleri güncelleyin:

### 1. default.sh

```bash
VEH="${VEHICLE_NAME:-autobot01}"
```

### 2. assignment3.launch

```xml
<arg name="veh" default="$(optenv VEHICLE_NAME autobot01)"/>
```

### 3. navigator_node.py

```python
ROBOT_NAME_DEFAULT = "autobot01"
```

---

## Build

```bash
dts devel build -f --arch arm32v7 -H ROBOTNAME.local
```

`ROBOTNAME` yerine kendi Duckiebot hostname’inizi yazın.

---

## Run

```bash
dts devel run -H ROBOTNAME.local
```

---

## Beklenen Çıktı

```text
Path sequence: N0 → N1 → N2 → N6 → N7 → N11 → N15
Total cost: 7.5
Goal Reached
```

---

## Sorun Giderme

### AprilTag Algılanmıyorsa

Varsayılan topic:

```text
/<robot_name>/apriltag_detector_node/detections
```

Kontrol etmek için:

```bash
rostopic list | grep -i april
```

Farklıysa şu parametreyi güncelle:

```python
~apriltag_detections_topic
```

Yanlış topic kullanılırsa robot hedefi bulamaz ve kendi etrafında dönebilir.

---

## Notlar

* ROS ortamının doğru kurulu olduğundan emin olun
* Duckiebot bağlantısını kontrol edin
* Kamera ve AprilTag pipeline düzgün çalışmalı
