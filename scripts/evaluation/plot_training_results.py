import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


src_dir = './runs/yolov8_box_parcel_second_stage'
paths_to_res = list(Path(src_dir).rglob("*.csv"))

models = []
for pt_res in paths_to_res:
    model = str(pt_res.parent.name)[:7]
    models.append(model)

colors = ['Red', 'DeepPink', 'Blue', 'LimeGreen', 'DarkTurquoise']
models2colors = dict(map(lambda i, j: (i, j), models, colors))


def plot_training_results(paths_to_res, plot_type='losses'):

    if plot_type == 'seg_losses':
        titles = ["Обучение (seg)", "Валидация (seg)"]
        y_title = "Значение функции потерь сегментации"
        legend_position = dict(yanchor="top", y=1, xanchor="center", x=0.96)
    elif plot_type == 'cls_losses':
        titles = ["Обучение (cls)", "Валидация (cls)"]
        y_title = "Значение функции потерь классификации"
        legend_position = dict(yanchor="top", y=1, xanchor="center", x=0.96)
    else:
        titles = ["mAP@50 (M)", "mAP@50:95 (M)"]
        y_title = "Значение метрики качества"
        legend_position = dict(yanchor="top", y=0.46, xanchor="center", x=0.96)

    fig = make_subplots(rows=1, cols=2, subplot_titles=titles)
    fig.update_annotations(font_size=22)

    for pt_res in paths_to_res:
        model = str(pt_res.parent.name)[:7]
        df = pd.read_csv(str(pt_res))
        df['epoch'] = df['                  epoch']

        if plot_type == 'seg_losses':
            df['train/seg_loss'] = df['         train/seg_loss'].astype(float)
            df['val/seg_loss'] = df['           val/seg_loss'].astype(float)

            fig.add_trace(go.Scatter(name=model, x=df['epoch'], y=df['train/seg_loss'],
                                     legendgroup=model, marker=dict(color=models2colors[model])),
                          row=1, col=1)
            fig.add_trace(go.Scatter(name=model, x=df['epoch'], y=df['val/seg_loss'],
                                     legendgroup=model, marker=dict(color=models2colors[model]),
                                     showlegend=False), row=1, col=2)

        elif plot_type == 'cls_losses':
            df['train/cls_loss'] = df['         train/cls_loss'].astype(float)
            df['val/cls_loss'] = df['           val/cls_loss'].astype(float)

            fig.add_trace(go.Scatter(name=model, x=df['epoch'], y=df['train/cls_loss'],
                                     legendgroup=model, marker=dict(color=models2colors[model])),
                          row=1, col=1)
            fig.add_trace(go.Scatter(name=model, x=df['epoch'], y=df['val/cls_loss'],
                                     legendgroup=model, marker=dict(color=models2colors[model]),
                                     showlegend=False), row=1, col=2)

        else:
            df['metrics/mAP50(M)'] = df['       metrics/mAP50(M)'].astype(float)
            df['metrics/mAP50-95(M)'] = df['    metrics/mAP50-95(M)'].astype(float)

            fig.add_trace(go.Scatter(name=model, x=df['epoch'], y=df['metrics/mAP50(M)'],
                                     legendgroup=model, marker=dict(color=models2colors[model])),
                          row=1, col=1)
            fig.add_trace(go.Scatter(name=model, x=df['epoch'], y=df['metrics/mAP50-95(M)'],
                                     legendgroup=model, marker=dict(color=models2colors[model]),
                                     showlegend=False), row=1, col=2)

    for col in [1, 2]:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='grey',
                         title_text="Номер эпохи", row=1, col=col)

    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='grey',
                     title_text=y_title, row=1, col=1)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='grey',
                     row=1, col=2)

    fig.update_layout(
        plot_bgcolor='white',
        legend_title_text='',
        height=600,
        legend=legend_position,
        font=dict(family='Times New Roman', size=18)
    )
    fig.show()


plot_training_results(paths_to_res, plot_type='seg_losses')
plot_training_results(paths_to_res, plot_type='cls_losses')
plot_training_results(paths_to_res, plot_type='metrics')
