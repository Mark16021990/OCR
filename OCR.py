import os
import json
import pytesseract
import cv2
import numpy as np
import re
import datetime
import shutil
import sys
import traceback

# Конфигурация Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Рабочие директории
WORK_DIR = r'C:\Users\Марк\Desktop\РАБОТА\Согаз Работа\Задачи\Сервис разпознования долкументов\OCR'
RESULTS_DIR = os.path.join(WORK_DIR, "Результат")
DEBUG_DIR = os.path.join(WORK_DIR, "debug")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Шаблоны для проверки данных
PASSPORT_SERIES_PATTERN = re.compile(r'\b\d{2}\s?\d{2}\s?\d{6}\b')
DATE_PATTERN = re.compile(r'\b\d{2}\.\d{2}\.\d{4}\b')
CODE_PATTERN = re.compile(r'\b\d{3}[-–]\d{3}\b')
GENDER_PATTERN = re.compile(r'\b(МУЖ|ЖЕН)\b')
NAME_PATTERN = re.compile(r'\b([А-ЯЁ]{2,})\s+([А-ЯЁ]{2,})\s+([А-ЯЁ]{2,})\b')
MRZ_PATTERN = re.compile(r'[A-Z0-9<]{9,}')

def load_image(file_path):
    """Загрузка изображения с улучшенной обработкой"""
    try:
        nparr = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            with open(file_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Ошибка загрузки изображения: {str(e)}")
        return None

def enhance_passport_image(img):
    """Улучшение качества изображения паспорта"""
    # Увеличение разрешения
    h, w = img.shape[:2]
    scale_factor = 3 if max(h, w) < 2000 else 1.5
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Улучшение контраста в LAB пространстве
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Удаление шума
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 15, 15, 9, 25)
    return denoised

def preprocess_for_ocr(image, is_mrz=False):
    """Предобработка для OCR с разными настройками для MRZ"""
    # Конвертация в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if is_mrz:
        # Для MRZ используем бинаризацию Оцу
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Увеличение резкости для MRZ
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(binary, -1, kernel)
        return sharpened
    else:
        # Для остальных областей используем адаптивную бинаризацию
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 91, 15
        )
        return binary

def extract_text_from_image(img, is_passport=False, psm=6, is_mrz=False):
    """Извлечение текста с выбором режима PSM и языков"""
    try:
        if is_passport:
            enhanced = enhance_passport_image(img)
            processed = preprocess_for_ocr(enhanced, is_mrz)
            
            if is_mrz:
                # Для MRZ используем английский язык и спец. символы
                config = f'--oem 3 --psm {psm} -l eng --tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
            else:
                # Для остальных областей - русский язык
                config = f'--oem 3 --psm {psm} -l rus'
            
            text = pytesseract.image_to_string(processed, config=config)
            return text.strip()
        else:
            processed = preprocess_for_ocr(img)
            config = '--oem 3 --psm 3 -l rus+eng'
            text = pytesseract.image_to_string(processed, config=config)
            return text.strip()
    except Exception as e:
        print(f"Ошибка OCR: {str(e)}")
        return ""

def extract_passport_fields(img):
    """Извлечение регионов паспорта РФ с улучшенной логикой"""
    h, w = img.shape[:2]
    
    # Верхняя часть (серия/номер, дата выдачи, код подразделения)
    top_region = img[0:int(h*0.25), 0:w]
    
    # Средняя часть (личные данные)
    middle_region = img[int(h*0.25):int(h*0.75), 0:w]
    
    # MRZ-зона (машиночитаемая зона)
    mrz_region = img[int(h*0.85):h, 0:w]
    
    # Сохранение регионов для отладки
    timestamp = datetime.datetime.now().strftime("%H%M%S%f")
    cv2.imwrite(os.path.join(DEBUG_DIR, f"top_region_{timestamp}.jpg"), top_region)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"middle_region_{timestamp}.jpg"), middle_region)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"mrz_region_{timestamp}.jpg"), mrz_region)
    
    return {
        "top": extract_text_from_image(top_region, True, 6),
        "middle": extract_text_from_image(middle_region, True, 3),
        "mrz": extract_text_from_image(mrz_region, True, 11, True)  # Специальная обработка для MRZ
    }

def parse_mrz(mrz_text):
    """Парсинг машиночитаемой зоны (MRZ)"""
    result = {}
    lines = [line.strip() for line in mrz_text.split('\n') if line.strip()]
    
    if len(lines) >= 2:
        # Первая строка MRZ
        line1 = lines[0].replace(' ', '').upper()
        if len(line1) > 5 and line1.startswith('P<'):
            # Формат: P<RUSLASTNAME<<FIRSTNAME<MIDDLENAME<<<<<<<<<
            country_code = line1[2:5]
            if country_code == 'RUS':
                result["Страна"] = "Россия"
            
            # Извлечение ФИО
            parts = line1[5:].split('<<', 1)
            if parts:
                # Фамилия
                result["Фамилия"] = parts[0].replace('<', ' ').strip()
                
                # Имя и отчество
                if len(parts) > 1:
                    name_parts = parts[1].split('<')
                    if name_parts:
                        result["Имя"] = name_parts[0].strip()
                        if len(name_parts) > 1:
                            result["Отчество"] = name_parts[1].strip()
        
        # Вторая строка MRZ
        line2 = lines[1].replace(' ', '')
        if len(line2) >= 28:
            # Формат: 1234567890RUS8001015M2101015<<<<<<<<<<<<<<<6
            
            # Серия и номер паспорта (первые 10 символов, но обычно 9 цифр + контрольная цифра)
            passport_serial = line2[0:9]
            if passport_serial.isdigit():
                result["Серия и номер паспорта"] = f"{passport_serial[0:2]} {passport_serial[2:4]} {passport_serial[4:]}"
            
            # Гражданство (3 буквы)
            citizenship = line2[10:13]
            if citizenship == 'RUS':
                result["Страна"] = "Россия"
            
            # Дата рождения (YYMMDD)
            dob = line2[13:19]
            if dob.isdigit():
                result["Дата рождения"] = f"{dob[4:6]}.{dob[2:4]}.19{dob[0:2]}"
            
            # Пол
            gender = line2[20]
            if gender == 'M':
                result["Пол"] = "МУЖ."
            elif gender == 'F':
                result["Пол"] = "ЖЕН."
            
            # Дата истечения срока (YYMMDD)
            expiry = line2[21:27]
            if expiry.isdigit():
                result["Срок действия"] = f"{expiry[4:6]}.{expiry[2:4]}.20{expiry[0:2]}"
    
    return result

def validate_date(date_str):
    """Проверка валидности даты"""
    try:
        if not date_str or len(date_str) != 10:
            return False
        day, month, year = map(int, date_str.split('.'))
        datetime.datetime(year, month, day)
        return True
    except:
        return False

def extract_place_of_birth(text):
    """Извлечение места рождения с улучшенной логикой"""
    # Поиск даты рождения
    dob_match = DATE_PATTERN.search(text)
    if dob_match:
        start_index = dob_match.end()
        remaining_text = text[start_index:]
        
        # Удаление ненужных символов
        clean_text = re.sub(r'[^А-ЯЁа-яё\s.,-]', '', remaining_text).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Удаление возможных цифр в конце
        clean_text = re.sub(r'\d+$', '', clean_text).strip()
        
        # Удаление лишних слов
        clean_text = re.sub(r'\b(г\.?р\.?|род\.?|место рождения)\b', '', clean_text, flags=re.IGNORECASE).strip()
        
        return clean_text
    return ""

def is_russian_passport(text):
    """Определение паспорта РФ с пониженным порогом"""
    keywords = [
        "паспорт", "россия", "фамилия", "имя", "отчество",
        "рождения", "выдан", "подразделения", "выдачи",
        "место", "пол", "гражданин", "удостоверение"
    ]
    text_lower = text.lower()
    return sum(1 for keyword in keywords if keyword in text_lower) >= 2

def parse_passport_data(text_fields):
    """Парсинг данных паспорта РФ"""
    result = {}
    top_text = text_fields.get("top", "")
    middle_text = text_fields.get("middle", "")
    mrz_text = text_fields.get("mrz", "")
    
    # Сохраняем распознанные тексты для отладки
    timestamp = datetime.datetime.now().strftime("%H%M%S%f")
    with open(os.path.join(DEBUG_DIR, f"top_text_{timestamp}.txt"), "w", encoding="utf-8") as f:
        f.write(top_text)
    with open(os.path.join(DEBUG_DIR, f"middle_text_{timestamp}.txt"), "w", encoding="utf-8") as f:
        f.write(middle_text)
    with open(os.path.join(DEBUG_DIR, f"mrz_text_{timestamp}.txt"), "w", encoding="utf-8") as f:
        f.write(mrz_text)

    # Стандартные значения
    result["Государство"] = "Российская Федерация"
    result["Орган, выдавший документ"] = "Федеральная миграционная служба"
    result["Страна"] = "Россия"
    result["Машиночитаемая зона (MRZ)"] = mrz_text.strip()

    # Парсинг MRZ
    mrz_data = parse_mrz(mrz_text)
    
    # Серия и номер паспорта (в первую очередь из MRZ)
    if "Серия и номер паспорта" in mrz_data:
        result["Серия и номер паспорта"] = mrz_data["Серия и номер паспорта"]
    else:
        serial_match = PASSPORT_SERIES_PATTERN.search(top_text)
        if serial_match:
            serial_str = re.sub(r'\s', '', serial_match.group(0))
            result["Серия и номер паспорта"] = f"{serial_str[:2]} {serial_str[2:4]} {serial_str[4:]}"

    # Дата выдачи и код подразделения
    dates = DATE_PATTERN.findall(top_text)
    if dates:
        for date in dates:
            if validate_date(date):
                result["Дата выдачи"] = date
                break
    
    codes = CODE_PATTERN.findall(top_text)
    if codes:
        result["Код подразделения"] = codes[0].replace("–", "-")

    # ФИО (в первую очередь из MRZ)
    if "Фамилия" in mrz_data:
        result["Фамилия"] = mrz_data["Фамилия"]
        result["Имя"] = mrz_data.get("Имя", "")
        result["Отчество"] = mrz_data.get("Отчество", "")
    else:
        name_match = NAME_PATTERN.search(middle_text)
        if name_match:
            result["Фамилия"] = name_match.group(1)
            result["Имя"] = name_match.group(2)
            result["Отчество"] = name_match.group(3)

    # Пол (в первую очередь из MRZ)
    if "Пол" in mrz_data:
        result["Пол"] = mrz_data["Пол"]
    else:
        gender_match = GENDER_PATTERN.search(middle_text)
        if gender_match:
            result["Пол"] = gender_match.group(1) + "."

    # Дата рождения (в первую очередь из MRZ)
    if "Дата рождения" in mrz_data:
        result["Дата рождения"] = mrz_data["Дата рождения"]
    else:
        dob_matches = DATE_PATTERN.findall(middle_text)
        if dob_matches:
            for dob in dob_matches:
                if validate_date(dob):
                    result["Дата рождения"] = dob
                    break

    # Место рождения
    place_text = extract_place_of_birth(middle_text)
    if place_text:
        # Очистка и форматирование места рождения
        place_text = re.sub(r'\s+', ' ', place_text).strip()
        result["Место рождения"] = place_text.upper()

    # Срок действия из MRZ
    if "Срок действия" in mrz_data:
        result["Срок действия"] = mrz_data["Срок действия"]

    return result

def process_passport(file_path):
    """Обработка паспорта РФ с улучшенной диагностикой"""
    try:
        img = load_image(file_path)
        if img is None:
            return {"error": "Не удалось загрузить изображение"}
        
        # Сохранение оригинального изображения
        timestamp = datetime.datetime.now().strftime("%H%M%S%f")
        cv2.imwrite(os.path.join(DEBUG_DIR, f"original_{timestamp}.jpg"), img)
        
        # Проверка типа документа
        full_text = extract_text_from_image(img, True, 3)
        with open(os.path.join(DEBUG_DIR, f"full_text_{timestamp}.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)
        
        # Более мягкая проверка на паспорт
        if not is_russian_passport(full_text):
            print("Предупреждение: Не все ключевые слова найдены, но продолжим обработку")
        
        # Извлечение данных
        text_fields = extract_passport_fields(img)
        passport_data = parse_passport_data(text_fields)
        
        # Формирование результата с группировкой по страницам
        result = {
            "Тип документа": "Паспорт РФ",
            "Имя файла": os.path.basename(file_path),
            "Время обработки": datetime.datetime.now().isoformat(),
            "Верхняя страница": {
                "Государство": passport_data.get("Государство", ""),
                "Орган, выдавший документ": passport_data.get("Орган, выдавший документ", ""),
                "Дата выдачи": passport_data.get("Дата выдачи", ""),
                "Код подразделения": passport_data.get("Код подразделения", ""),
                "Серия и номер паспорта": passport_data.get("Серия и номер паспорта", "")
            },
            "Нижняя страница": {
                "Фамилия": passport_data.get("Фамилия", ""),
                "Имя": passport_data.get("Имя", ""),
                "Отчество": passport_data.get("Отчество", ""),
                "Пол": passport_data.get("Пол", ""),
                "Дата рождения": passport_data.get("Дата рождения", ""),
                "Место рождения": passport_data.get("Место рождения", "")
            },
            "Дополнительно": {
                "Машиночитаемая зона (MRZ)": passport_data.get("Машиночитаемая зона (MRZ)", ""),
                "Страна": passport_data.get("Страна", ""),
                "Срок действия": passport_data.get("Срок действия", "")
            }
        }
        return result
    except Exception as e:
        return {
            "error": f"Ошибка обработки: {str(e)}", 
            "traceback": traceback.format_exc()
        }

def save_results(data, filename):
    """Сохранение результатов в JSON"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{os.path.splitext(filename)[0]}_{timestamp}.json"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return result_path
    except Exception as e:
        print(f"Ошибка сохранения: {str(e)}")
        return None

def main(file_path):
    """Основная функция обработки файла"""
    if not os.path.exists(file_path):
        return {"error": "Файл не найден"}
    
    try:
        # Обработка документа
        result = process_passport(file_path)
        
        # Сохранение результатов
        result_path = save_results(result, os.path.basename(file_path))
        if result_path and "error" not in result:
            result["Путь к результату"] = result_path
        
        return result
    except Exception as e:
        error_info = {
            "error": f"Ошибка обработки: {str(e)}",
            "traceback": traceback.format_exc()
        }
        error_path = save_results(error_info, f"error_{os.path.basename(file_path)}")
        if error_path:
            error_info["Путь к ошибке"] = error_path
        return error_info

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = os.path.abspath(sys.argv[1])
        result = main(file_path)
        
        print("\nРезультат распознавания:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if "Путь к результату" in result:
            print(f"\nРезультат сохранен в: {result['Путь к результату']}")
    else:
        print("Использование: python скрипт.py <путь_к_файлу>")