import gradio as gr
import PIL.Image as Image
import time
from ultralytics import YOLO
import multiprocessing
from detector import Detector
import os
from functions import *

# Initialize the Detector model
model = Detector()

# Function to train the YOLO model
def train_yolo(model_name, time_input):
    """Load the YOLO model and train it on the user's dataset."""
    model = YOLO("runs/train/yolo11x.pt")
    model.train(
        data="dataset/data.yaml",
        time=time_input,
        imgsz=640,
        batch=-1,
        name=model_name,
    )

# Function to predict objects in images using the YOLO model
def predict_image(imgs, text_id):
    """Predicts objects in an image using the YOLO model."""
    _imgs = [Image.open(img).resize((640, 640)) for img in imgs]
    results = model.detect(_imgs)
    im = model.plot(_imgs)
    return im

# Function to delete specified detections and re-display images
def dell(imgs, ids):
    """Delete detections by ID and re-display images."""
    ids = ids.split(",")
    model.delete_bbox_by_id(ids)
    _imgs = [Image.open(img).resize((640, 640)) for img in imgs]
    im = model.plot(_imgs)
    return im

# Function to process confirmed detections and save data
def done(l_id, imgs):
    """Confirm detections, save images and annotations."""
    dataset_dir = "dataset"
    data = model.get_report(l_id)
    save_laptop_data(data, "res.csv")
    
    for i, img in enumerate(imgs):
        result = model.results[i] if i < len(model.results) else []
        img_name = str(len(os.listdir(os.path.join(dataset_dir, "train/images"))))
        
        img = Image.open(img)
        img = img.resize((640, 640))
        img.save(os.path.join(dataset_dir, "train/images", f"{img_name}.jpg"))
        save_annotations(result, os.path.join(dataset_dir, "train/labels", f"{img_name}.txt"))

# Function to generate and download the report
def down_report():
    """Generate statistics report from saved data and return PDF filename."""
    t = time.time()
    header, data, counts, average = generate_statistics("res.csv", 0, t)
    pdf_filename = 'statistics_report.pdf'
    save_statistics_to_pdf(header, data, counts, average, pdf_filename)
    return pdf_filename

# Function to start or stop the training process
def start_stop_train(model_name_input, time_input, button_text):
    """Start or stop the training of the YOLO model."""
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

# Create Gradio interface for inference
with gr.Blocks() as demo:
    g = gr.Gallery(type="pil", label="Result")
    grrr = gr.File(
        file_count="multiple",
        file_types=["image", "directory"],
        label="Upload Images or Directory",
    )
    l_id = gr.Textbox(label="Input Laptop ID")
    iface = gr.Interface(
        fn=predict_image,
        inputs=[grrr, l_id],
        outputs=[g],
        title="Детектор дефектов",
        description="Загрузите изображение или несколько изображений для детекции",
    )

    # Additional UI components
    input_string = gr.Textbox(label="ID детекций для удаления (1,2,3,4)")
    process_btn = gr.Button("Удалить выбранные детекции")
    done_btn = gr.Button("Подтвердить")
    down_btn = gr.Button("Выгрузить отчет")
    download_output = gr.File(label="Скачать отчет")

    # Connect buttons to functions
    process_btn.click(fn=dell, inputs=[grrr, input_string], outputs=[g])
    done_btn.click(fn=done, inputs=[l_id, grrr], outputs=None)
    down_btn.click(fn=down_report, inputs=None, outputs=download_output)

# Create Gradio interface for training
with gr.Blocks() as iface2:
    button = gr.Button("Начать обучение")
    model_name_input = gr.Dropdown(
        choices=["Битых пикселей", "Наличия болтов", "Сколов и царапин"],
        label="Модель для детекции",
    )
    time_input = gr.Number(label="Время обучения в часах", minimum=0.5, value=5)
    output = gr.Textbox(label="Output")

    # Handle training input and connect it to function
    button.click(
        fn=start_stop_train,
        inputs=[model_name_input, time_input, button],
        outputs=[output, button],
    )

# Create a tabbed interface to switch between inference and training
if __name__ == "__main__":
    demo = gr.TabbedInterface([demo, iface2], ["Детекция дефектов", "Обучение моделей"])
    demo.launch(share=False, show_error=True)
