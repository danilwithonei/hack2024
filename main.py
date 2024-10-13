import gradio as gr
import PIL.Image as Image
import time
from ultralytics import YOLO
import multiprocessing
from detector import Detector
import os
from functions import *

# Function to train the YOLO model
def train_yolo(model_name, time_input):
    # Load the pre-trained model
    model = YOLO("runs/train/yolo11x.pt")
    # Train the model on the user's dataset
    model.train(
        data="dataset/data.yaml",
        time=time_input,
        imgsz=640,
        batch=-1,
        name=model_name,
    )


# Function to predict objects in an image using the YOLO model
def predict_image(imgs, text_id):
    """Predicts objects in an image using a YOLO11 model with adjustable confidence and IOU thresholds."""
    res = []
    _imgs = []
    for img in imgs:
        img = Image.open(img)
        img = img.resize((640, 640))
        _imgs.append(img)
    results = model.detect(_imgs)
    im = model.plot(_imgs)

    return im


def dell(imgs, ids):
    ids = ids.split(",")
    res = []
    _imgs = []
    model.delete_bbox_by_id(ids)
    for img in imgs:
        img = Image.open(img)
        img = img.resize((640, 640))
        _imgs.append(img)
    im = model.plot(_imgs)
    return im

model = Detector()

# Create the Gradio interface for inference
g = gr.Gallery(type="pil", label="Result")
grrr = gr.File(
    file_count="multiple",
    file_types=["image", "directory"],
    label="Upload Images or Directory",
)
l_id = gr.Textbox(label="Input Laptop ID")
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        grrr,
        l_id,
    ],
    outputs=[g],
    title="Детектор дефектов",
    description="Загругите изображение или несколо изображений",
)
def done(l_id,imgs):
    dataset_dir = "dataset"
    data = model.get_report(l_id)
    save_laptop_data(data,"res.csv")
    for i,img in enumerate(imgs):
        if i >= len(model.results):
            result = []
        else:
            result = model.results[i]

        img_name = str(len(os.listdir(os.path.join(dataset_dir,"train/images"))))
        img = Image.open(img)
        img = img.resize((640, 640))
        img.save(os.path.join(dataset_dir,"train/images",f"{img_name}.jpg"))
        save_annotations(result,os.path.join(dataset_dir,"train/labels",f"{img_name}.txt"))

def down_report():
    t = time.time()
    header, data, counts, average = generate_statistics("res.csv", 0, t)
    pdf_filename = 'statistics_report.pdf'
    save_statistics_to_pdf(header, data, counts, average, pdf_filename)
    return 'statistics_report.pdf'

with gr.Blocks() as demo:
    iface.render()
    with gr.Column():
        input_string = gr.Textbox(label="ID детекций для удаления (1,2,3,4)")
        process_btn = gr.Button("Удалить выбранные детекции")
        done_btn = gr.Button("Подтвердить")
        down_btn = gr.Button("Выгрузить отчет")
        download_output = gr.File(label="Скачать отчет")

    # Connect the button to the additional function
    process_btn.click(fn=dell, inputs=[grrr,input_string], outputs=[g])
    done_btn.click(fn=done,inputs=[l_id,grrr],outputs=None)
    down_btn.click(fn=down_report,inputs=None,outputs=download_output)


# Function to start or stop the training process
def start_stop_train(model_name_input, time_input, button_text):
    global training_process
    if button_text == "Начать обучение":
        training_process = multiprocessing.Process(
            target=train_yolo, args=(model_name_input, time_input)
        )
        training_process.start()
        return "Обучение началось", "Остановить обучение"
    else:
        training_process.terminate()
        return "Обучение завершено", "Начать обучение"


# Create the Gradio interface for training
with gr.Blocks() as iface2:
    # Create a button with an initial text
    button = gr.Button("Начать обучение")

    # Define the inputs
    model_name_input = gr.Dropdown(
        choices=["Битых пикселей", "Наличия болтов", "Сколов и царапин"],
        label="Модель для детекции",
    )
    time_input = gr.Number(label="Время обучения в часах", minimum=0.5, value=5)

    # Define the output
    output = gr.Textbox(label="Output")

    # Define the function to handle inputs and connect it to the inputs and output
    button.click(
        fn=start_stop_train,
        inputs=[model_name_input, time_input, button],
        outputs=[output, button],
    )

# Create a tabbed interface to switch between inference and training

if __name__ == "__main__":
    demo = gr.TabbedInterface([demo, iface2], ["Детекция дефектов", "Обучение моделей"])
    demo.launch(share=False, show_error=True)
