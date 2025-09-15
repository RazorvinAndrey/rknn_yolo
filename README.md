# Инструкция по запуску различных нейросетевых моделей семейства YOLO на NPU Rockchip (RKNN)

## 1  Настройка окружения на ПК Linux x86 (не одноплатный компьютер)

### Скачивание репозиториев для работы с RKNN

    # Create the 'Projects' folder
    mkdir Projects

    # Switch to this directory
    cd Projects

    # Download the RKNN-Toolkit2 repository
    git clone https://github.com/airockchip/rknn-toolkit2.git --depth 1

    # Download the RKNN Model Zoo repository
    git clone https://github.com/airockchip/rknn_model_zoo.git --depth 1

    # Скачивание этого репозитория (необходимо разрешение)
    git clone https://github.com/RazorvinAndrey/rknn_yolo.git

Структура каталога:

    Projects
    ├── rknn-toolkit2
    │ ├── doc
    │ ├── rknn-toolkit2
    │ │ ├── packages
    │ │ ├── docker
    │ │ └── ...
    │ ├── rknpu2
    │ │ ├── runtime
    │ │ └── ...
    │ └── ...
    ├── rknn_model_zoo
    | ├── datasets
    | ├── examples
    | └── ...
    └── rknn_yolo
      ├── media
      ├── models
      ├── yolo_6
      ├── yolo_8
      └── ...

