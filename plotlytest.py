import plotly.express as px
import pandas as pd

dt = pd.DataFrame(
    data={
        "x": [1, 2, 3, 4, 5, 6],
        "y": [10, 11, 12, 13, 34, 65],
        "cont": ["a", "b", "c", "d", "a", "b"],
        "pop": [3, 7, 8, 9, 2, 3],
        "country": ["india", "usa", "uk", "australia", "india", "usa"],
    }
)
# df = pd.read_excel(r"C:\Users\harsh\Downloads\data.xlsx")
# print(df.query("year==2007"))
# df.query("year==2007").to_excel(r"C:\Users\harsh\Downloads\output.xlsx")
fig = px.scatter(
    # df.query("year==2007"),
    dt,
    # x="gdpPercap",
    # y="lifeExp",
    x="x",
    y="y",
    size="pop",
    # color="continent",
    color="cont",
    hover_name="country",
    # log_x=True,
    # size_max=60,
)
fig.show()
# import plotly.graph_objects as go


# fig = go.Figure(
#     data=[
#         go.Scatter(
#             x=[1, 2, 3, 4, 5, 6, 7, 8],
#             y=[10, 11, 12, 13, 34, 65],
#             mode="markers",
#             marker=dict(
#                 color=[
#                     # "rgb(93, 164, 214)",
#                     # "rgb(255, 144, 14)",
#                     # "rgb(44, 160, 101)",
#                     # "rgb(255, 65, 54)",

#                 ],
#                 opacity=[1, 0.8, 0.6, 0.4],
#                 size=[40, 60, 80, 100],
#             ),
#         )
#     ]
# )

# fig.show()
