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