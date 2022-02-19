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
  'pair0x32', 'pair1x32', 'pair2x32', 'pair3x32', 'pair4x32', 'pair5x32', 'pair6x32', 'pair7x32',
  'pair8x32', 'pair9x32', 'pair10x32', 'pair11x32', 'pair12x32', 'pair13x32', 'pair14x32', 'pair15x32',
  'pair16x32', 'pair17x32', 'pair18x32', 'pair19x32', 'pair20x32', 'pair21x32', 'pair22x32', 'pair23x32',
  'pair24x32', 'pair25x32', 'pair26x32', 'pair27x32', 'pair28x32', 'pair29x32', 'pair30x32', 'pair31x32',
  'pair32x32', 'pair33x32', 'pair34x32', 'pair35x32', 'pair36x32', 'pair37x32', 'pair38x32', 'pair39x32',
  'pair40x32', 'pair41x32', 'pair42x32', 'pair43x32', 'pair44x32', 'pair45x32', 'pair46x32', 'pair47x32',
  'pair48x32', 'pair49x32', 'pair50x32', 'pair51x32', 'pair52x32', 'pair53x32', 'pair54x32', 'pair55x32',
  'pair56x32', 'pair57x32', 'pair58x32', 'pair59x32', 'pair60x32', 'pair61x32', 'pair62x32', 'pair63x32',
];

var horizontal = [
  'pair+0', 'pair+1', 'pair+2', 'pair+3', 'pair+4', 'pair+5', 'pair+6', 'pair+7',
  'pair+8', 'pair+9', 'pair+10', 'pair+11', 'pair+12', 'pair+13', 'pair+14', 'pair+15',
  'pair+16', 'pair+17', 'pair+18', 'pair+19', 'pair+20', 'pair+21', 'pair+22', 'pair+23',
  'pair+24', 'pair+25', 'pair+26', 'pair+27', 'pair+28', 'pair+29', 'pair+30', 'pair+31',
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

var ws = new WebSocket("ws://59.78.194.240:8080/cc1024")
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