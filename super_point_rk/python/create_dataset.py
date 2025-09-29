import os
import sys
import argparse
from PIL import Image, ImageOps

def resize_image_with_padding(input_path, output_path, target_size):
    """
    Изменяет размер изображения с сохранением пропорций и добавляет черные поля при необходимости
    """
    try:
        # Открываем изображение
        with Image.open(input_path) as img:
            # Конвертируем в RGB если нужно
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGBA')
            else:
                img = img.convert('RGB')
            
            # Получаем текущие размеры
            original_width, original_height = img.size
            target_width, target_height = target_size
            
            # Вычисляем соотношения сторон
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            
            # Выбираем минимальное соотношение для сохранения пропорций
            ratio = min(width_ratio, height_ratio)
            
            # Вычисляем новые размеры
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # Изменяем размер изображения
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Создаем новое изображение с черным фоном
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            
            # Вычисляем позицию для вставки (по центру)
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Вставляем измененное изображение в центр
            new_img.paste(resized_img, (x_offset, y_offset))
            
            # Сохраняем результат
            new_img.save(output_path)
            return True
            
    except Exception as e:
        print(f"Ошибка при обработке {input_path}: {e}")
        return False

def process_images(input_folder, output_folder, target_size):
    """
    Обрабатывает все изображения в папке
    """
    # Создаем выходную папку если ее нет
    os.makedirs(output_folder, exist_ok=True)
    
    # Список для хранения путей к обработанным изображениям
    processed_paths = []
    
    # Поддерживаемые форматы изображений
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # Обрабатываем все файлы в папке
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"Обработка: {filename}")
            
            if resize_image_with_padding(input_path, output_path, target_size):
                # Получаем абсолютный путь
                abs_output_path = os.path.abspath(output_path)
                processed_paths.append(abs_output_path)
                print(f"Успешно: {filename} -> {target_size}")
            else:
                print(f"Ошибка: {filename}")
    
    return processed_paths

def create_dataset_file(processed_paths, output_folder):
    """
    Создает файл dataset.txt с абсолютными путями
    """
    dataset_file = os.path.join(output_folder, "dataset.txt")
    
    with open(dataset_file, 'w', encoding='utf-8') as f:
        for path in processed_paths:
            f.write(path + '\n')
    
    print(f"Файл dataset.txt создан: {dataset_file}")
    return dataset_file

def parse_size(size_str):
    """
    Парсит строку размера в формате WIDTHxHEIGHT
    """
    try:
        width, height = map(int, size_str.lower().split('x'))
        return (width, height)
    except:
        raise argparse.ArgumentTypeError("Размер должен быть в формате WIDTHxHEIGHT (например: 800x600)")

def main():
    parser = argparse.ArgumentParser(description='Изменение размера изображений с сохранением пропорций')
    parser.add_argument('input_folder', help='Путь к папке с исходными изображениями')
    parser.add_argument('output_folder', help='Путь к папке для сохранения результатов')
    parser.add_argument('size', type=parse_size, help='Целевой размер в формате WIDTHxHEIGHT')
    
    args = parser.parse_args()
    
    # Проверяем существование входной папки
    if not os.path.exists(args.input_folder):
        print(f"Ошибка: Папка {args.input_folder} не существует")
        sys.exit(1)
    
    print(f"Обработка изображений из: {args.input_folder}")
    print(f"Сохранение в: {args.output_folder}")
    print(f"Целевой размер: {args.size[0]}x{args.size[1]}")
    print("-" * 50)
    
    # Обрабатываем изображения
    processed_paths = process_images(args.input_folder, args.output_folder, args.size)
    
    # Создаем файл dataset.txt
    if processed_paths:
        dataset_file = create_dataset_file(processed_paths, args.output_folder)
        print(f"\nОбработано изображений: {len(processed_paths)}")
        print(f"Файл с путями: {dataset_file}")
    else:
        print("Не найдено изображений для обработки")

if __name__ == "__main__":
    main()
