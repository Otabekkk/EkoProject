import qrcode

def generateCode(userName: str, user_id: int):
    qr = qrcode.QRCode(
        version = 1,
        error_correction = qrcode.constants.ERROR_CORRECT_L,
        box_size = 10,
        border = 4,
    )

    qr.add_data(f'http://localhost:5001/scan/{user_id}')
    qr.make(fit = True)
    img = qr.make_image(fill = 'black', back_color = 'white')
    qr_code_path = f"static/qr_codes/{user_id}.png"
    img.save(qr_code_path)
    return qr_code_path