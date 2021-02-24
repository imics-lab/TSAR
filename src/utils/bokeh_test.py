#Author: Gentry Atkinson
#Organization: Texas University
#Data: 10 December, 2020
#Just a toy file to figure out bokeh

from bokeh.plotting import figure, output_file, show, ColumnDataSource
import numpy as np
from bokeh.models import Button, CustomJS
from bokeh.layouts import column
from bokeh.io import curdoc

def button_handler():
    print("That's a click")

if __name__ == "__main__":
    X = np.arange(0, 6.28, 0.1)
    y = np.sin(X)
    output_file("bokeh_test.html")

    CustomJS(code="console.log('Custom JS ran', this.toString())")

    p = figure(
        tools="pan,box_zoom,reset,save",
        y_axis_type="linear", y_range=[-1, 1], title="Only A Test",
        x_axis_label='X', y_axis_label='Sin(X)'
    )
    p.line(X, y, legend_label="y=sin(x)")

    button = Button(label="Button", button_type="success")
    button.js_on_click(CustomJS(code="console.log('button: click!', this.toString())"))
    #button.js_on_click(CustomJS(code=callback))
    # button.js_on_click(CustomJS(code="""
    #     var data = source.data;
    #     value1=data['x'];
    #     var out = "";
    #     var file = new Blob([out], {type: 'text/plain'});
    #     var elem = window.document.createElement('a');
    #     elem.href = window.URL.createObjectURL(file);
    #     elem.download = 'button_clicks.txt';
    #     document.body.appendChild(elem);
    #     elem.click();
    #     document.body.removeChild(elem);
    #     """))

    show(column(p,button))
