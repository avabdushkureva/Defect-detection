from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def count_labels_per_each_class(src_dir: str):
    paths_to_images = [i for i in Path(src_dir).rglob("*") if i.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    paths_to_labels = list(Path(src_dir).rglob("*.txt"))

    num_instances_hole = 0
    num_instances_opened = 0
    num_instances_dent = 0

    for pt_label in tqdm(paths_to_labels):

        with open(pt_label, "r") as txt_file:
            lines = txt_file.readlines()
            for line in lines:
                cl = line.strip().split(" ")[0]
                cl = eval(cl)

                if cl == 0:
                    num_instances_hole += 1
                elif cl == 1:
                    num_instances_opened += 1
                elif cl == 2:
                    num_instances_dent += 1

    return len(paths_to_images), len(paths_to_labels), num_instances_hole, num_instances_opened, num_instances_dent


num_images, num_labels, hole, opened, dent = count_labels_per_each_class(src_dir="./datasets/Second_stage")

data_1 = ['Изображения', 'Посылки без дефектов', 'Дефекты']
numbers_1 = [num_images, num_images-num_labels, hole+opened+dent]
data_2 = ['Отверстие', 'Открытая коробка', 'Вмятина']
numbers_2 = [hole, opened, dent]

fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Общая статистика", "Баланс классов"), shared_yaxes=True
)
fig.update_annotations(font_size=22)
fig.add_trace(go.Bar(x=data_1, y=numbers_1, marker=dict(color='#7B68EE'), showlegend=False,
                     text=numbers_1), row=1, col=1)
fig.add_trace(go.Bar(x=data_2, y=numbers_2, marker=dict(color=['LightBlue', 'LemonChiffon', 'Plum']),
                     showlegend=False, text=numbers_2), row=1, col=2)

fig.update_layout(
    plot_bgcolor='white',
    yaxis_title="Количество",
    font=dict(family='Times New Roman', size=18),
    yaxis_range=[0, 5400],
    height=600, width=1300
)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='grey')
fig.update_traces(textposition='outside', width=0.6)

fig.show()
