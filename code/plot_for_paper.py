import json
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from matplotlib.pyplot import yscale

folder = "for_plot"
files = [f"{folder}/good1.json",f"{folder}/good2.json",f"{folder}/good3.json",f"{folder}/good4.json" , f"{folder}/bad1.json", f"{folder}/bad2.json", f"{folder}/bad3.json"]

data_per_colors = {}
xs = []
ys = []
for f in files:
    data_per_colors[f]={}
    data_per_color = data_per_colors[f]
    with open(f, 'r') as file:
        data = json.load(file)
        data = data["data"]
        for i, d in enumerate(data):
            name = d["name"] if "name" in d else str(i)
            x = d["x"]
            y = d["y"]
            xs.extend(x)
            ys.extend(y)

            data_per_color[name] = (x, y)

    count = 1
    for color, (x, y) in data_per_color.items():
        if color not in [str(i) for i in range(0,4)]: continue
        if "0" in data_per_color:
            color = str(int(color)+1)
        plt.scatter(x, y, label=str(count), s=100)
        count+=1

    ret, triangle = cv.minEnclosingTriangle(
        np.round(np.unique(np.array([[x, y] for x, y in zip(xs, ys)]).astype(np.float32), axis=0), 4)
    )
    triangle = triangle.squeeze()

    # add line for plt to plt
    plt.plot([triangle[0][0], triangle[1][0]], [triangle[0][1], triangle[1][1]], c="black")
    plt.plot([triangle[1][0], triangle[2][0]], [triangle[1][1], triangle[2][1]], c="black")
    plt.plot([triangle[2][0], triangle[0][0]], [triangle[2][1], triangle[0][1]], c="black")

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Scatter Plot by Color {f}')
    plt.xticks([])  # Remove x-axis numbers
    plt.yticks([])  # Remove y-axis numbers
    plt.title('')  # Remove title
    plt.legend()
    plt.legend(prop={'size': 19})
    plt.savefig(f'./plots/{f}_scatter_plot.png')  # Save the plot as a PDF file
    plt.show()

f = f"{folder}/sdp.json"
data_per_colors[f]={}
data_per_color = data_per_colors[f]
with open(f, 'r') as file:
    data = json.load(file)
    data = data["data"]
    for i, d in enumerate(data):
        name = d["marker"]["color"]
        x = d["x"]
        y = d["y"]
        for j, c in enumerate(name):
            if str(c) not in data_per_color:
                data_per_color[str(c)]= ([],[])
            data_per_color[str(c)][0].append(x[j])
            data_per_color[str(c)][1].append(y[j])

    count = 1
    for color, (x, y) in data_per_color.items():
        if color not in [str(i) for i in range(0,4)]: continue
        if "0" in data_per_color:
            color = str(int(color)+1)
        plt.scatter(x, y, label=str(count), s=100)
        count += 1
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Scatter Plot by Color {f}')
    plt.xticks([])  # Remove x-axis numbers
    plt.yticks([])  # Remove y-axis numbers
    plt.title('')  # Remove title
    plt.legend()
    plt.legend(prop={'size': 19})
    plt.savefig(f'./plots/{f}_scatter_plot.pdf')  # Save the plot as a PDF file
    plt.show()


colors = ["#00a041","#ff7f00", "#0477c4"]
f = f"{folder}/support.json"
data_per_colors[f]={}
data_per_color = data_per_colors[f]
xs = []
ys = []
with open(f, 'r') as file:
    data = json.load(file)
    data = data["data"]
    for i, d in enumerate(data):
        name = d["name"] if "name" in d else str(i)
        x = d["x"]
        y = d["y"]
        data_per_color[name] = (x, y)
        xs.extend(x)
        ys.extend(y)

count = 1
for color in ["-10", "-11", "-12"] + ["3", "2", "1"]:
    x,y = data_per_color[color]
    if color not in [str(i) for i in range(1,4)] + ['-10', '-11', '-12']: continue
    label = None
    # if color == '-10':
    #     label = "1"
    if color in [str(i) for i in range(1,4)]:
        label = color
        if label == "1":
            label = "3"
        elif label == "3":
            label = "1"

    c = colors[int(color)-1] if color in [str(i) for i in range(1,4)] else colors[2]
    print(color, c, label)
    plt.scatter(x, y, label=label, marker="o" if color in [str(i) for i in range(0,4)] else "x",
                c=c, s=100)
    if color in [str(i) for i in range(1,4)]:
        count+=1

ret, triangle = cv.minEnclosingTriangle(
    np.round(np.unique(np.array([[x, y] for x, y in zip(xs, ys)]).astype(np.float32), axis=0), 4)
)
triangle = triangle.squeeze()

# add line for plt to plt
plt.plot([triangle[0][0], triangle[1][0]], [triangle[0][1], triangle[1][1]], c="black", linestyle='--')
plt.plot([triangle[1][0], triangle[2][0]], [triangle[1][1], triangle[2][1]], c="black", linestyle='--')
plt.plot([triangle[2][0], triangle[0][0]], [triangle[2][1], triangle[0][1]], c="black", linestyle='--')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xticks([])  # Remove x-axis numbers
plt.yticks([])  # Remove y-axis numbers
plt.title('')  # Remove title
plt.legend()
plt.legend(prop={'size': 19})
plt.savefig(f'./plots/{f}_scatter_plot.png')  # Save the plot as a PDF file
plt.show()


