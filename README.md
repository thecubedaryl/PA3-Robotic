# Duckietown Assignment 3

## Proje Özeti

Bu proje, bir Duckiebot üzerinde Duckietown Assignment 3 çalışmasını gerçekleştirmek için şunları kullanır:

- `N0` noktasından `N15` noktasına en kısa yolu hesaplamak için A* pathfinding
- Bir sonraki hedef düğümü algılamak için ARTag (AprilTag) tabanlı konumlama
- Duckiebot'u planlanan yol boyunca sürmek için ROS tabanlı hareket kontrolü

Navigator düğümü yolu önce çevrimdışı olarak hesaplar, ardından gerçek zamanlı AprilTag algılamalarını kullanarak robotu düğümden düğüme ilerletir ve son hedefe ulaştırır.

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

### Ana Dosyalar

- `assignment3/packages/assignment3/src/astar.py`
  A* algoritmasını uygular ve planlanan yolu ile toplam maliyeti ekrana yazdırır.

- `assignment3/packages/assignment3/src/navigator_node.py`
  Assignment 3 için ana ROS düğümüdür. Yolu yükler, AprilTag algılamalarına abone olur ve Duckiebot için hareket komutları yayınlar.

- `assignment3/packages/assignment3/launch/assignment3.launch`
  Navigator düğümünü başlatan ve robot adını parametre olarak geçiren ROS launch dosyasıdır.

- `assignment3/launchers/default.sh`
  `dts devel run` tarafından kullanılan Duckietown launcher scriptidir.

## Robot Adı Nasıl Değiştirilir

Duckiebot adınız `autobot01` değilse, robot adını aşağıdaki 3 yerde güncelleyin:

### 1. `assignment3/launchers/default.sh`

Şu satırı değiştirin:

```bash
VEH="${VEHICLE_NAME:-autobot01}"
```

Gerekirse `autobot01` yerine kendi robot adınızı yazın.

### 2. `assignment3/packages/assignment3/launch/assignment3.launch`

Şu satırı değiştirin:

```xml
<arg name="veh" default="$(optenv VEHICLE_NAME autobot01)"/>
```

Buradaki `autobot01` değerini kendi Duckiebot adınız ile değiştirin.

### 3. `assignment3/packages/assignment3/src/navigator_node.py`

Şu satırı değiştirin:

```python
ROBOT_NAME_DEFAULT = "autobot01"
```

`autobot01` yerine kendi robot adınızı yazın.

## Build

Şu komutu çalıştırın:

```bash
dts devel build -f --arch arm32v7 -H ROBOTNAME.local
```

Buradaki `ROBOTNAME` kısmını Duckiebot hostname değeri ile değiştirin.

## Run

Şu komutu çalıştırın:

```bash
dts devel run -H ROBOTNAME.local
```

## Beklenen Terminal Çıktısı

Düğüm başarıyla başladığında aşağıdakine benzer bir çıktı görmelisiniz:

```text
Path sequence: N0 → N1 → N2 → N6 → N7 → N11 → N15
Total cost: 7.5
Goal Reached
```

Hareket süresi birebir aynı olmayabilir, ancak yol sırası, toplam maliyet ve sonunda `Goal Reached` mesajı görünmelidir.

## Sorun Giderme

### ARTag Topic Yanlışsa

Duckiebot tepki vermiyorsa veya etiketleri hiç algılamıyorsa, AprilTag topic adı yanlış olabilir.

Navigator varsayılan olarak şu topic'e abone olur:

```text
/<robot_name>/apriltag_detector_node/detections
```

Robot üzerinde doğru topic adını kontrol etmek için şu komutu çalıştırın:

```bash
rostopic list | grep -i april
```

Eğer sizin sisteminiz farklı bir AprilTag detections topic'i kullanıyorsa, düğüme verilen şu parametreyi güncelleyin:

```python
~apriltag_detections_topic
```

Bu değişikliği `navigator_node.py` içinde yapabilir veya launch yapılandırması üzerinden verebilirsiniz.

Topic adı yanlışsa robot geçerli tag algılaması alamaz. Bu yüzden hedefe gitmek yerine sürekli arama davranışı gösterebilir ve kendi etrafında dönebilir.
