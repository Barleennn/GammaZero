import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from aiogram.fsm.context import FSMContext
import asyncio
from PIL import Image
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Укажите путь к изображению объекта для поиска
small_image_path = 'image/1.png'

# Определяем состояние для обработки
class Form(StatesGroup):
    waiting_for_pdf = State()

# Инициализация бота
API_TOKEN = '7611578010:AAHt2uEA-nHSCcxqFebhyzCsvhqzHZr84tM'
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

def process_slide(large_image):
    logger.info("Начало обработки слайда")
    # Создаем копию изображения, чтобы можно было его модифицировать
    large_image = large_image.copy()
    
    # Получаем размеры шаблона
    small_image = cv2.imread(small_image_path)
    h, w = small_image.shape[:2]

    # Шаблонное сопоставление
    result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
    
    # Установим порог для определения совпадений
    threshold = 0.8
    yloc, xloc = np.where(result >= threshold)

    # Если найдено совпадение
    if len(xloc) > 0:
        logger.info(f"Найдено {len(xloc)} совпадений на слайде")
        for (x, y) in zip(xloc, yloc):
            # Создаем маску для объекта
            mask = np.zeros(large_image.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255  # Обозначаем область объекта

            # Получаем цвета границ объекта
            border_colors = []
            border_distance = 5  # Расстояние от границы для получения цветов

            for dy in range(-border_distance, border_distance + 1):
                for dx in range(-border_distance, border_distance + 1):
                    if abs(dy) == border_distance or abs(dx) == border_distance:
                        border_x = min(max(x + dx * w // 2, 0), large_image.shape[1] - 1)
                        border_y = min(max(y + dy * h // 2, 0), large_image.shape[0] - 1)
                        border_colors.append(large_image[border_y, border_x])

            # Находим медианный цвет границ
            if len(border_colors) > 0:
                avg_color = np.median(border_colors, axis=0).astype(int)

                # Закрашиваем объект средним цветом границ
                large_image[mask == 255] = avg_color
    else:
        logger.info("Совпадений на слайде не найдено")

    logger.info("Завершение обработки слайда")
    return large_image

@dp.message(Command("start"))
async def start_command(message: Message, state: FSMContext):
    logger.info(f"Пользователь {message.from_user.id} запустил бота")
    await message.reply("Отправьте мне PDF-файл для обработки.")
    await state.set_state(Form.waiting_for_pdf)

@dp.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    user_id = message.from_user.id
    logger.info(f"Получен документ от пользователя {user_id}")
    
    input_pdf_path = f'input_{user_id}.pdf'
    output_pdf_path = f'output_{user_id}.pdf'
    
    try:
        # Проверяем формат файла
        if not message.document.file_name.lower().endswith('.pdf'):
            await message.reply("Пожалуйста, отправьте файл в формате PDF.")
            return

        # Сохраняем PDF файл
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, input_pdf_path)
        logger.info(f"PDF файл сохранен для пользователя {user_id}")

        # Открываем PDF и обрабатываем каждый слайд
        logger.info(f"Начало обработки PDF для пользователя {user_id}")
        
        with fitz.open(input_pdf_path) as doc, fitz.open() as output_doc:
            for page_num, page in enumerate(doc):
                logger.info(f"Обработка страницы {page_num + 1} для пользователя {user_id}")
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                processed_img = process_slide(img)
                
                # Конвертируем в RGB и сохраняем как временный PNG
                img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                temp_png_path = f'temp_{user_id}_{page_num}.png'
                cv2.imwrite(temp_png_path, img_rgb)
                
                try:
                    # Создаем новую страницу в выходном документе
                    new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
                    
                    # Вставляем PNG на новую страницу
                    new_page.insert_image(new_page.rect, filename=temp_png_path)
                finally:
                    # Удаляем временный PNG файл
                    if os.path.exists(temp_png_path):
                        os.remove(temp_png_path)
                        logger.info(f"Временный PNG файл удален: {temp_png_path}")

            # Сохраняем готовый PDF
            output_doc.save(output_pdf_path)
            logger.info(f"Обработанный PDF сохранен для пользователя {user_id}")

        # Отправляем готовый PDF
        doc_to_send = FSInputFile(output_pdf_path)
        await message.reply_document(doc_to_send)
        logger.info(f"Обработанный PDF отправлен пользователю {user_id}")

    except Exception as e:
        logger.error(f"Ошибка при обработке файла для пользователя {user_id}: {str(e)}")
        await message.reply(f"Произошла ошибка при обработке файла: {str(e)}")
    finally:
        # Очистка временных файлов
        for file_path in [input_pdf_path, output_pdf_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Временный файл удален: {file_path}")
            except Exception as e:
                logger.error(f"Ош бика при удалении временного файла: {str(e)}")

        await state.clear()
        logger.info(f"Состояние очищено для пользователя {user_id}")

async def main():
    logger.info("Запуск бота")
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
