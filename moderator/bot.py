import logging
import requests
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums import ParseMode

import asyncio
from aiogram.client.default import DefaultBotProperties

TOKEN = "7993090177:AAF3i1HxiyG_WFiGIQfwcYId4M63toBgggs"
API_URL = "http://localhost:50001/scan/"

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()


MODERATORS = [5693398070]


def decode_qr_code(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    decoded_objects = decode(img)
    if decoded_objects:
        return decoded_objects[0].data.decode("utf-8")
    return None


@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer("Привет! Отправь мне QR-код, чтобы начислить баллы.")

# Хэндлер обработки фото с QR-кодом
@dp.message(lambda message: message.photo)
async def process_qr_photo(message: Message):
    if message.from_user.id not in MODERATORS:
        await message.answer("⛔ У вас нет прав для начисления баллов.")
        return

    photo = message.photo[-1]
    photo_file = await bot.download(photo)

    qr_code = decode_qr_code(photo_file.read())
    
    if qr_code:
        await message.answer(f"📡 QR-код распознан: {qr_code}")
        await send_points(message, qr_code)
    else:
        await message.answer("❌ Не удалось распознать QR-код.")


@dp.message(lambda message: message.text.startswith("http"))
async def process_qr_text(message: Message):
    if message.from_user.id not in MODERATORS:
        await message.answer("⛔ У вас нет прав для начисления баллов.")
        return

    qr_code = message.text.split("=")[-1]
    await message.answer(f"📡 Код получен: {qr_code}")
    await send_points(message, qr_code)


async def send_points(message: Message, qr_code: str):
    url = qr_code
    response = requests.get(url)
    print(url)

    if response.status_code == 200:
        await message.answer(f"✅ Баллы успешно начислены пользователю!")
    else:
        await message.answer(f"❌ Ошибка при начислении баллов для пользователя.")


async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())