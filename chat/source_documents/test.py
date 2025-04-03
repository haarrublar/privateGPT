import pdfplumber

# Open the PDF file
with pdfplumber.open('./LisethBarbosa-TesisMaster.pdf') as pdf, open('./result.txt', "w", encoding="utf-8") as txt_file:
    for page in pdf.pages:
        text = page.extract_text()
        if text:  # Check if text exists on the page
            txt_file.write(text + "\n\n")  # Write text to file with spacingt extracted text from each page