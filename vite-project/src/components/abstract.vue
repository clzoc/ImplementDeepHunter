<template>
  <el-row :gutter="10">
    <el-col :span="2"></el-col>
    <el-col :span="4">
      <div
        style="border-style: solid; border-radius: 8px; border-width: 3px; border-color: #409eff; height: 40px; line-height: 40px;"
      >current layer coverage:</div>
    </el-col>
    <el-col :span="6">
      <div
        style="border-style: solid; border-radius: 22px; border-width: 3px; border-color: #2c3e50; height: 40px;"
      >
        <el-progress :text-inside="true" :stroke-width="40.5" :percentage="coverage.valueOf()"></el-progress>
      </div>
    </el-col>
    <el-col :span="1"></el-col>
    <el-col :span="4">
      <div
        style="border-style: solid; border-radius: 8px; border-width: 3px; border-color: #409eff; height: 40px; line-height: 40px;"
      >global coverage:</div>
    </el-col>
    <el-col :span="6">
      <div
        style="border-style: solid; border-radius: 22px; border-width: 3px; border-color: #2c3e50; height: 40px;"
      >
        <el-progress :text-inside="true" :stroke-width="40.5" :percentage="0"></el-progress>
      </div>
    </el-col>
  </el-row>
  <div ref="myChart" style="height: 1000px; width: 100%;"></div>
</template>

<script setup>
import { ref, onMounted } from "vue"
import * as echarts from 'echarts'
import 'echarts-gl';

var hours = ['Neu0', 'Neu1', 'Neu2', 'Neu3', 'Neu4', 'Neu5', 'Neu6',
  'Neu7', 'Neu8', 'Neu9', 'Neu10', 'Neu11', 'Neu12',
  'Neu13', 'Neu14', 'Neu15', 'Neu16', 'Neu17', 'Neu18',
  'Neu19', 'Neu20', 'Neu21', 'Neu22', 'Neu23', 'Neu24',
  'Neu25', 'Neu26', 'Neu27', 'Neu28', 'Neu29', 'Neu30',
  'Neu31', 'Neu32', 'Neu33', 'Neu34', 'Neu35', 'Neu36',
  'Neu37', 'Neu38', 'Neu39', 'Neu40', 'Neu41', 'Neu42',
  'Neu43', 'Neu44', 'Neu45', 'Neu46', 'Neu47', 'Neu48',
  'Neu49', 'Neu50', 'Neu51', 'Neu52', 'Neu53', 'Neu54',
  'Neu55', 'Neu56', 'Neu57', 'Neu58', 'Neu59', 'Neu60',
  'Neu61', 'Neu62', 'Neu63',
];
var days = ['Neu0', 'Neu1', 'Neu2', 'Neu3', 'Neu4', 'Neu5', 'Neu6',
  'Neu7', 'Neu8', 'Neu9', 'Neu10', 'Neu11', 'Neu12',
  'Neu13', 'Neu14', 'Neu15', 'Neu16', 'Neu17', 'Neu18',
  'Neu19', 'Neu20', 'Neu21', 'Neu22', 'Neu23', 'Neu24',
  'Neu25', 'Neu26', 'Neu27', 'Neu28', 'Neu29', 'Neu30',
  'Neu31', 'Neu32', 'Neu33', 'Neu34', 'Neu35', 'Neu36',
  'Neu37', 'Neu38', 'Neu39', 'Neu40', 'Neu41', 'Neu42',
  'Neu43', 'Neu44', 'Neu45', 'Neu46', 'Neu47', 'Neu48',
  'Neu49', 'Neu50', 'Neu51', 'Neu52', 'Neu53', 'Neu54',
  'Neu55', 'Neu56', 'Neu57', 'Neu58', 'Neu59', 'Neu60',
  'Neu61', 'Neu62', 'Neu63',
];

let data = null

let myChart = ref()
let chartInstance = null
let option = null
let signal = 1

var coverage = ref(0)

var ws = new WebSocket("ws://59.78.194.133:8000/inf")
ws.onopen = function () {
  ws.send(signal)
}
ws.onmessage = function (event) {
  data = JSON.parse(event.data)
  coverage.value = data[1] * 100
  draw_chart()
};

function initChart() {
  chartInstance = echarts.init(myChart.value, 'vintage')
}

function draw_chart() {
  option = {
    tooltip: {},
    visualMap: {
      min: 0,
      max: 4,
      left: '3%',
      bottom: '40%',
      inRange: {
        // color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        color: ['#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
      }
    },
    xAxis3D: {
      type: 'category',
      data: hours
    },
    yAxis3D: {
      type: 'category',
      data: days
    },
    zAxis3D: {
      type: 'value'
    },
    grid3D: {
      boxWidth: 200,
      boxDepth: 200,
      viewControl: {
        // projection: 'orthographic'
      },
      light: {
        main: {
          intensity: 1.2,
          shadow: true
        },
        ambient: {
          intensity: 0.3
        }
      }
    },
    series: [{
      type: 'bar3D',
      data: data[0].map(function (item) {
        return {
          value: [item[1], item[0], item[2]],
        }
      }),
      shading: 'lambert',

      label: {
        fontSize: 16,
        borderWidth: 1
      },

      emphasis: {
        label: {
          fontSize: 20,
          color: '#900'
        },
        itemStyle: {
          color: '#900'
        }
      }
    }]
  }
  chartInstance.setOption(option)
}


onMounted(() => {
  initChart()
})


</script>

<style scoped>
</style>