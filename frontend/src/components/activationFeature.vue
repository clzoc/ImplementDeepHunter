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

  <el-pagination
    background
    layout="prev, pager, next"
    :total="320"
    @current-change="handleCurrentChange"
  ></el-pagination>
  <div ref="requestDate" style="height: 1000px; width: 100%;"></div>
</template>

<script setup>
import "../assets/font/font.css"
import { ref, onMounted } from "vue"
import * as echarts from 'echarts'

let requestDate = ref()
let store_data = null
let chartInstance = null
let option = null

let batch = 1

let former = 0
let latter = 1

var coverage = ref(0)

var ws = new WebSocket("ws://59.78.194.240:8080/snowball")
ws.onopen = function () {
  ws.send(batch)
}

ws.onmessage = function (event) {
  store_data = JSON.parse(event.data)
  coverage.value = store_data[1] * 100
  draw_chart()
};

function handleCurrentChange(current) {
  former = (current - 1) * 2 + 0
  latter = (current - 1) * 2 + 1
  batch = current
  ws.send(batch)
}

// store_data = [
//   [
//     [0, 0, 0],
//     [0, 1, 1],
//     [0, 2, 2],
//     [0, 3, 3],
//     [0, 4, 4],
//     [1, 0, 5],
//     [1, 1, 6],
//     [1, 2, 7],
//     [1, 3, 8],
//     [1, 4, 9],
//     [2, 0, 10],
//     [2, 1, 11],
//     [2, 2, 12],
//     [2, 3, 13],
//     [2, 4, 14],
//     [3, 0, 15],
//     [3, 1, 16],
//     [3, 2, 17],
//     [3, 3, 18],
//     [3, 4, 19],
//     [4, 0, 20],
//     [4, 1, 21],
//     [4, 2, 22],
//     [4, 3, 23],
//     [4, 4, 24]
//   ],

//   [
//     [0, 0, 0],
//     [0, 1, 1],
//     [0, 2, 2],
//     [0, 3, 3],
//     [0, 4, 4],
//     [1, 0, 5],
//     [1, 1, 6],
//     [1, 2, 7],
//     [1, 3, 8],
//     [1, 4, 9],
//     [2, 0, 10],
//     [2, 1, 11],
//     [2, 2, 12],
//     [2, 3, 13],
//     [2, 4, 14],
//     [3, 0, 15],
//     [3, 1, 16],
//     [3, 2, 17],
//     [3, 3, 18],
//     [3, 4, 19],
//     [4, 0, 20],
//     [4, 1, 21],
//     [4, 2, 22],
//     [4, 3, 23],
//     [4, 4, 24]
//   ]
// ]

function initChart() {
  chartInstance = echarts.init(requestDate.value, 'vintage')
}

function draw_chart() {
  option = {
    title: [
      {
        text: "neuron " + former,
        left: '28.5%',
        top: '1%',
        textAlign: 'center',
      },
      {
        text: "neuron " + latter,
        left: '74.5%',
        top: '1%',
        textAlign: 'center'
      }
    ],
    textStyle:
    {
      fontFamily: 'Kaushan'
    },
    grid: [
      {
        left: '8%', top: '5%', width: '896px', height: '896px'
      },
      {
        left: '54%', top: '5%', width: '896px', height: '896px'
      },
    ],
    tooltip: {},
    xAxis: [
      {
        gridIndex: 0,
        type: 'category',
        show: false,
      },
      {
        gridIndex: 1,
        type: 'category',
        show: false,
      },
    ],
    yAxis: [
      {
        gridIndex: 0,
        type: 'category',
        show: false,
        inverse: true,
      },
      {
        gridIndex: 1,
        type: 'category',
        show: false,
        inverse: true,
      },
    ],
    visualMap: {
      type: 'continuous',
      // type: 'piecewise',
      min: -1,
      max: 5,
      calculable: true,
      realtime: false,
      orient: 'vertical',
      show: true,
      itemHeight: 180,
      left: '3%',
      bottom: '40%',
      // splitNumber: 1000,
      // precision: 5,
      inRange: {
        color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
      }
    },
    series: [
      {
        name: 'first',
        type: 'heatmap',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: store_data[0][0],
        progressive: 1000
      },
      {
        name: 'second',
        type: 'heatmap',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: store_data[0][1],
        progressive: 1000
      },
    ]
  };
  chartInstance.setOption(option)
}

onMounted(() => {
  initChart()
})

</script>


<style scoped>
/* div {
  margin: 0 auto;
  height: 1000px;
  width: 100%;
} */
</style>