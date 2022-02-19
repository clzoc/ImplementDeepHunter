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
        <el-progress
          :text-inside="true"
          :stroke-width="40.5"
          :percentage="currentCoverage.valueOf()"
        ></el-progress>
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
        <el-progress
          :text-inside="true"
          :stroke-width="40.5"
          :percentage="globalCoverage.valueOf()"
        ></el-progress>
      </div>
    </el-col>
  </el-row>
  <div ref="myChart" style="height: 1000px; width: 100%;"></div>
</template>

<script setup>
import { ref, onMounted, reactive, watch } from "vue"
import * as echarts from 'echarts'
import 'echarts-gl';
import { onBeforeRouteLeave, onBeforeRouteUpdate, useRoute } from "vue-router";

var vertical = [
  'Neu+0', 'Neu+1', 'Neu+2', 'Neu+3', 'Neu+4', 'Neu+5', 'Neu+6', 'Neu+7',
  'Neu+8', 'Neu+9', 'Neu+10', 'Neu+11', 'Neu+12', 'Neu+13', 'Neu+14', 'Neu+15',
  'Neu+16', 'Neu+17', 'Neu+18', 'Neu+19', 'Neu+20', 'Neu+21', 'Neu+22', 'Neu+23',
  'Neu+24', 'Neu+25', 'Neu+26', 'Neu+27', 'Neu+28', 'Neu+29', 'Neu+30', 'Neu+31',
  'Neu+32', 'Neu+33', 'Neu+34', 'Neu+35', 'Neu+36', 'Neu+37', 'Neu+38', 'Neu+39',
  'Neu+40', 'Neu+41', 'Neu+42', 'Neu+43', 'Neu+44', 'Neu+45', 'Neu+46', 'Neu+47',
  'Neu+48', 'Neu+49', 'Neu+50', 'Neu+51', 'Neu+52', 'Neu+53', 'Neu+54', 'Neu+55',
  'Neu+56', 'Neu+57', 'Neu+58', 'Neu+59', 'Neu+60', 'Neu+61', 'Neu+62', 'Neu+63',
];

var horizontal = [
  'Neu0x64', 'Neu1x64', 'Neu2x64', 'Neu3x64', 'Neu4x64', 'Neu5x64', 'Neu6x64', 'Neu7x64',
  'Neu8x64', 'Neu9x64', 'Neu10x64', 'Neu11x64', 'Neu12x64', 'Neu13x64', 'Neu14x64', 'Neu15x64',
  'Neu16x64', 'Neu17x64', 'Neu18x64', 'Neu19x64', 'Neu20x64', 'Neu21x64', 'Neu22x64', 'Neu23x64',
  'Neu24x64', 'Neu25x64', 'Neu26x64', 'Neu27x64', 'Neu28x64', 'Neu29x64', 'Neu30x64', 'Neu31x64',
];

var layer_index = ref('')

const route = useRoute()
watch(() => route.query, (newvalue, oldvalue) => {
  layer_index.value = route.query.layer
  ws.send(layer_index.value)
})

let data = null

let myChart = ref()
let chartInstance = null
let option = null

var currentCoverage = ref(0)
var globalCoverage = ref(0)

var ws = new WebSocket("ws://59.78.194.240:8080/nc")
ws.onopen = function () {
  // ws.addEventListener
  // ws.binaryType
  // ws.dispatchEvent
  // ws.removeEventListener
  ws.send(layer_index.value)
}
ws.onmessage = function (event) {
  data = JSON.parse(event.data)
  currentCoverage.value = data[0] * 100
  globalCoverage.value = data[1] * 100
  draw_chart()
  ws.send(layer_index.value)
};

function initChart() {
  chartInstance = echarts.init(myChart.value, 'vintage')
}

function draw_chart() {
  option = {
    tooltip: {},
    visualMap: {
      min: -10,
      max: 10,
      left: '3%',
      bottom: '40%',
      inRange: {
        color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        // color: ['#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
      }
    },
    xAxis3D: {
      type: 'category',
      data: horizontal
    },
    yAxis3D: {
      type: 'category',
      data: vertical
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
      data: data[2].map(function (item) {
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
  layer_index.value = route.query.layer
})

</script>

<style scoped>
</style>