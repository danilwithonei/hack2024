import csv
import os
from fpdf import FPDF
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np


def save_laptop_data(data, save_file):
    '''
    Функция сохранения данных о ноутбуке в общий файл
    '''

    mode = 'a' if os.path.exists(save_file) else 'w'
    with open(save_file, mode=mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if mode == 'w':
            writer.writeheader()
        writer.writerow(data)


def generate_statistics(load_file, date_from, date_to):
    '''
    Функция подсчета статистики по дефектам
    '''

    data = []

    with open(load_file, mode='r', newline='', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        header = csv_reader.fieldnames

        counts, average = dict(), dict()
        for element in header:
            if element != 'id' and element != 'date':
                counts[element], average[element] = 0, 0

        for row in csv_reader:
            data.append([])
            t = 0
            for element, value in row.items():
                data[-1].append(value)
                if element == 'date':
                    t = float(value)
                else:
                    if element != 'id' and date_from <= t and date_to >= t:
                        counts[element] += int(value)
                        average[element] = (average[element] + int(value)) / 2 if average[element] > 0 else int(value)

    return header, data, counts, average


def generate_pie(counts, average):
    '''
    Функция для создания диаграмм распределения дефектов
    '''

    plot_counts = []
    plot_average = []

    for defect in counts.keys():
        plot_counts.append(counts[defect])
        plot_average.append(average[defect])

    plt.pie(np.array(plot_average), labels=counts.keys(), autopct=lambda x: '{:.2f}'.format(x*np.array(plot_average).sum()/100.0))
    plt.savefig("average.png", format='png', bbox_inches="tight")
    plt.cla()
    plt.pie(np.array(plot_counts), labels=counts.keys(), autopct=lambda x: '{:.0f}'.format(x*np.array(plot_counts).sum()/100.0))
    plt.savefig("counts.png", format='png', bbox_inches="tight")


def save_statistics_to_pdf(header, data, counts, average, pdf_file):
    '''
    Формирование файла отчета по статистике в формате PDF
    '''

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)

    # Добавление заголовка
    pdf.cell(200, 10, txt="Статистика по дефектам", ln=True, align='C')
    
    # Добавление пирогов
    generate_pie(counts, average)
    pdf.cell(110, 10, "Среднее значение", ln=1, align='C')
    pdf.image("average.png", x=10, y=30, w=100)

    pdf.set_xy(90, 20)
    pdf.cell(130, 10, "Количество", ln=1, align='C') 
    pdf.image("counts.png", x=110, y=30, w=100) 
    pdf.set_xy(90, 110)

    pdf.cell(0, 10, txt="", ln=True)

    col_widths = [25 for i in range(len(header))] 
    col_widths[1] = 55
    col_widths[-1] = 35
    pdf.set_fill_color(230, 230, 230)  

    # Добавление текстов по статистике
    for defect in counts.keys():
        text = "{}: Количество = {}, Среднее количество = {}".format(defect, 
                                                     str(round(counts[defect], 2)), 
                                                     str(round(average[defect], 2)))
        pdf.cell(0, 10, txt=text, ln=True)
    
    pdf.cell(0, 10, txt="", ln=True)

    # Добавление таблицы
    for i, col_name in enumerate(header):
        pdf.cell(col_widths[i], 10, col_name, 1, 0, 'C', 1)

    pdf.ln()
    
    pdf.set_font('DejaVu', '', 12)
    for row in data:
        for i, element in enumerate(row):
            if i == 1:
                element = datetime.datetime.fromtimestamp(float(element)).strftime('%c')
            elif i>= 2:
                element = 'Да' if int(element) > 0 else 'Нет'
            pdf.cell(col_widths[i], 10, element, 1, 0, 'C')
        pdf.ln()

    pdf.output(pdf_file)


def save_annotations(data, filename):
    '''
    Функция сохранения разметки изображений в формате для дообучения моделей нейронных сетей
    '''
    
    with open(filename, 'w') as f:
        for element in data:
            f.write('{} {} {} {} {}\n'.format(int(element[1]), 
                                            round(element[3][0], 4), round(element[3][1], 4), 
                                            round(element[3][2], 4), round(element[3][3]), 4))


if __name__ == '__main__':
    # Примеры использования функций 

    import random
    filename = 'tmp.csv'

    for i in range(100):
        data = {'id' : random.randint(10000, 99999), 'date': time.time(), 
                'царапина' : random.randint(0, 1), 
                'скол' : random.randint(0, 1), 
                'замок' : random.randint(0, 1), 
                'битый пиксель' : random.randint(0, 1), }

        save_laptop_data(data, filename)

    t = time.time()
    header, data, counts, average = generate_statistics(filename, 0, t)

    pdf_filename = 'statistics_report.pdf'
    save_statistics_to_pdf(header, data, counts, average, pdf_filename)

    data = [[np.array([     191.57,      50.118,      231.63,      75.082], dtype=np.float32), 
             np.array(          2, dtype=np.float32), 
             np.array(    0.26656, dtype=np.float32), 
             np.array([    0.33063,    0.097813,    0.062598,    0.039007], dtype=np.float32), 
             'normal'],
             [np.array([     191.57,      50.118,      231.63,      75.082], dtype=np.float32), 
             np.array(          2, dtype=np.float32), 
             np.array(    0.26656, dtype=np.float32), 
             np.array([    0.33063,    0.097813,    0.062598,    0.039007], dtype=np.float32), 
             'coca-cola normal']]

    save_annotations(data, '1.txt')