#!/usr/bin/python3.5
from urllib.parse import urldefrag
from bokeh.models import LinearAxis, Range1d, CustomJS, HoverTool, BoxSelectTool
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
import json
import os
import logging
import coloredlogs
coloredlogs.install(level='DEBUG')

class DrawWaterfall():
    def __init__(self, jsonFile, outputFile, load_time):
        self.json_file = jsonFile
        with open(self.json_file) as data_file:
            self.data = json.load(data_file)[1:-3]
        # end_time = data[-1][1]['endTime'] + 500
        self.end_time = load_time * 1.1
        self.y_range = len(self.data) + 10
        self.line_width = 4
        output_file(outputFile)
        self.yr = Range1d(start=self.y_range, end=0)
        self.xr = Range1d(start=0, end=1.05 * self.end_time)
        hover = HoverTool(
            tooltips="""
                                  <div style='padding: 3px; width: 500px; word-break: break-all; word-wrap: break-word; text-align: left;'>
                                      <div>
                                          <div>
                                              <span style="font-weight: bold; font-size: 9px;">@desc</span>
                                          </div>
                                      </div>
                                      <div>
                                          <div>
                                              <span style=" font-size: 8px;">@o_url</span>
                                          </div>
                                      </div>
                                      <div>
                                          <div>
                                              <span style="font-size: 9px;">@o_size</span>
                                          </div>

                                      </div>
                                      <div>
                                          <div>
                                              <span style="font-size: 11px;">@o_stime</span>
                                          </div>

                                      </div>
                                      <div>
                                          <div>
                                              <span style="font-size: 11px;">@o_etime</span>
                                          </div>

                                      </div>

                                      <div>
                                          <div>
                                              <span style="font-size: 11px;">@o_time</span>
                                          </div>

                                      </div>
                                  </div>
                                  """
        )
        self.p = figure(plot_width=1250, plot_height=2100, tools=[hover, 'save,pan,wheel_zoom,box_zoom,reset,resize'],
                        y_range=self.yr,
                        x_range=self.xr, x_axis_location="above")
        # p.ygrid.grid_line_color = None
        self.p.xaxis.axis_label = 'Time (ms)'
        self.p.xaxis.axis_label_text_align = 'left'
        self.p.xaxis.axis_label_text_color = "#c8c8c8"
        self.p.xaxis.axis_label_text_font_size = '10pt'
        self.p.xaxis.axis_line_color = '#c8c8c8'
        self.p.xaxis.major_tick_line_color = '#c8c8c8'
        self.p.xaxis.major_label_text_color = '#c8c8c8'
        self.p.xaxis.major_label_text_align = 'left'
        self.p.xaxis.major_label_text_font_size = '10pt'
        self.p.xaxis.minor_tick_line_color = '#c8c8c8'
        self.p.xaxis.minor_tick_out = 0
        self.p.xgrid.grid_line_alpha = 0.5
        self.p.ygrid.grid_line_color = None
        self.p.yaxis.visible = False
        self.javascript_type_list = ['application/x-javascript', 'application/javascript', 'application/ecmascript',
                                     'text/javascript', 'text/ecmascript', 'application/json', 'javascript/text']
        self.css_type_list = ['text/css', 'css/text']
        self.text_type_list = ['evalhtml', 'text/html', 'text/plain', 'text/xml']
        self.colormap = dict(ctext='#2757ae', dtext="#a8c5f7", cjs="#c9780e", djs='#e8ae61', ccss="#13bd0d",
                             dcss='#8ae887',
                             cother="#eb5bc0", dother='#eb5bc0', img='#c79efa')

    def draw_from_json(self):
        for _index, _event in enumerate(self.data):
            if not _event['id'] == 'Deps':
                for _obj in _event['objs']:
                    _nodeId = _obj[0]
                    _nodeData = _obj[1]
                    try:
                        _startTime = round(_nodeData['startTime'], 2)
                    except:
                        print(_nodeData, _nodeData)
                        continue
                    try:
                        _endTime = round(_nodeData['endTime'], 2)
                    except:
                        print(_nodeId, _nodeData)
                        continue
                    _duration = round(_endTime - _startTime, 2)
                    ##########################################################################################
                    # Network
                    ##########################################################################################
                    if _nodeId.startswith('Network'):
                        if 'transferSize' in _nodeData:
                            _transferSize = _nodeData['transferSize']
                        else:
                            _transferSize = 0
                        _url = _nodeData['url']
                        _mimeType = _nodeData['mimeType']
                        y_index = (_index + 1)
                        if _mimeType in self.text_type_list:
                            color = self.colormap['dtext']
                        elif _mimeType in self.css_type_list:
                            color = self.colormap['dcss']
                        elif _mimeType in self.javascript_type_list:
                            color = self.colormap['djs']
                        elif _mimeType.startswith('image'):
                            color = self.colormap['img']
                        else:
                            color = self.colormap['dother']
                        _mimeType = _nodeId + ': ' + _nodeData['mimeType']
                        source = ColumnDataSource(
                            data=dict(
                                x=[_startTime, _endTime],
                                y=[y_index, y_index],
                                desc=[_mimeType, _mimeType],
                                o_url=[_url, _url],
                                o_size=[_transferSize, _transferSize],
                                o_stime=['s: ' + str(_startTime) + ' ms', 's: ' + str(_startTime) + ' ms'],
                                o_etime=['e: ' + str(_endTime) + ' ms', 'e: ' + str(_endTime) + ' ms'],
                                o_time=['dur: ' + str(_duration) + ' ms', 'dur: ' + str(_duration) + ' ms']
                            ))
                        r = self.p.line('x', 'y', source=source,
                                   line_color=color,
                                   line_width=self.line_width, line_cap='round', name='myline')
                    ##########################################################################################
                    # Loading
                    ##########################################################################################
                    elif _nodeId.startswith('Loading'):
                        _desc = _nodeData['name'] + ': ' + _nodeId
                        _url = ' '
                        _styleSheetUrl = ' '
                        if _nodeData['name'] == 'ParseHTML' and 'url' in _nodeData:
                            if _nodeData['url'] is not None:

                                _url = _nodeData['url']
                                y_index = _index + 1
                                color = self.colormap['ctext']
                            else:
                                continue
                        elif _nodeData['name'] == 'ParseAuthorStyleSheet' and 'styleSheetUrl' in _nodeData:
                            if _nodeData['styleSheetUrl'] is not None:
                                _styleSheetUrl = _nodeData['styleSheetUrl']
                                y_index = _index + 1
                                color = self.colormap['ccss']
                            else:
                                continue
                        source = ColumnDataSource(
                            data=dict(
                                x=[_startTime, _endTime],
                                y=[y_index, y_index],
                                desc=[_desc, _desc],
                                o_url=[_url, _url],
                                o_size=[_styleSheetUrl, _styleSheetUrl],
                                o_stime=['s: ' + str(_startTime) + ' ms', 's: ' + str(_startTime) + ' ms'],
                                o_etime=['e: ' + str(_endTime) + ' ms', 'e: ' + str(_endTime) + ' ms'],
                                o_time=['dur: ' + str(_duration) + ' ms', 'dur: ' + str(_duration) + ' ms']
                            ))
                        r = self.p.line('x', 'y', source=source,
                                   line_color=color,
                                   line_width=self.line_width, line_cap='round', name='myline')
                    ##########################################################################################
                    # Scripting
                    ##########################################################################################
                    elif _nodeId.startswith('Scripting'):
                        _url = _nodeData['url']
                        _desc = _nodeId
                        color = self.colormap['cjs']
                        y_index = _index + 1
                        source = ColumnDataSource(
                            data=dict(
                                x=[_startTime, _endTime],
                                y=[y_index, y_index],
                                desc=[_desc, _desc],
                                o_url=[_url, _url],
                                o_size=['Scripting', 'Scripting'],
                                o_stime=['s: ' + str(_startTime) + ' ms', 's: ' + str(_startTime) + ' ms'],
                                o_etime=['e: ' + str(_endTime) + ' ms', 'e: ' + str(_endTime) + ' ms'],
                                o_time=['dur: ' + str(_duration) + ' ms', 'dur: ' + str(_duration) + ' ms']
                            ))
                        r = self.p.line('x', 'y', source=source,
                                   line_color=color,
                                   line_width=self.line_width, line_cap='round', name='myline')

    def showPlot(self):
        show(self.p)


_experiment_dir = '/home/jnejati/PLTSpeed/desktop_b5d100-mo'
_plot_dir = '/var/plots/desktop_b5d100-mo'
def clear_folder(folder):
    if os.path.isdir(folder):
            for root, dirs, l_files in os.walk(folder):
                for f in l_files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
    else:
        os.makedirs(folder)
clear_folder(_plot_dir)

for _site_dir in os.listdir(_experiment_dir):
    _site_dir = os.path.join(_experiment_dir, _site_dir)
    _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
    for _run_no in _runs:
        _run_dir = os.path.join(_site_dir, _run_no)
        _analysis_dir = os.path.join(_run_dir, 'analysis')
        for _file in os.listdir(_analysis_dir):
            if _file.endswith('out'):
                _analysis_file = os.path.join(_analysis_dir, _file)
                with open(_analysis_file) as _f:
                    _data = json.load(_f)
                    _load_time = _data[0]['load']
                logging.info('Generating plot for ' + _site_dir + ' --' + _run_no)
                _plot_file = os.path.join(_plot_dir, str(_file) + '.html')
                _plot = DrawWaterfall(_analysis_file, _plot_file, _load_time)
                _plot.draw_from_json()
                _plot.showPlot()
