<template>
  <!-- <img alt="logo" src="./assets/scene.jpg" /> -->
  <!-- <HelloWorld msg="This Time" /> -->
  <!-- <br /> -->

  <el-row :gutter="10">
    <el-col :span="2">
      <div class="grid-content bg-purple">
        <router-link to="/">Home Page</router-link>
      </div>
    </el-col>

    <el-col :span="3">
      <div class="grid-content bg-purple">Select Model</div>
    </el-col>

    <el-col :span="2">
      <div class="custom1">
        <el-select popper-class="custom2" v-model="value" clearable placeholder="neural network">
          <el-option
            v-for="item in options"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          ></el-option>
        </el-select>
      </div>
    </el-col>

    <el-col :span="3">
      <div class="grid-content bg-purple">Select Coverage</div>
    </el-col>

    <el-col :span="2">
      <div class="custom1">
        <el-select
          popper-class="custom2"
          v-model="second_value"
          clearable
          placeholder="coverage kind"
        >
          <el-option
            v-for="item in second_options"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          ></el-option>
        </el-select>
      </div>
    </el-col>

    <el-col :span="3">
      <div class="grid-content bg-purple">Select Layer</div>
    </el-col>

    <el-col :span="2">
      <div class="custom1">
        <el-select popper-class="custom2" v-model="third_value" clearable placeholder="layer index">
          <el-option
            v-for="item in third_options"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          ></el-option>
        </el-select>
      </div>
    </el-col>

    <el-col :span="3">
      <div class="grid-content bg-purple">Select Display</div>
    </el-col>

    <el-col :span="2">
      <div class="custom1">
        <el-select
          popper-class="custom2"
          v-model="forth_value"
          clearable
          placeholder="display type"
        >
          <el-option
            v-for="item in forth_options"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          ></el-option>
        </el-select>
      </div>
    </el-col>

    <el-col :span="2">
      <div class="grid-content bg-purple">
        <router-link :to="{ path: '/' + route_name.valueOf() }">Go</router-link>
      </div>
    </el-col>
  </el-row>

  <router-view></router-view>
  <!-- <router-view /> -->
</template>

<script setup>
// import HelloWorld from './components/HelloWorld.vue'
import "./assets/font/font.css"
import { ref, watch } from 'vue'

let options = [
  {
    value: 'item0',
    label: 'Resnet50'
  },
  {
    value: 'item1',
    label: 'YoloV5'
  }
]
let value = ref('')


let second_options = [
  {
    value: 'item0',
    label: 'neuron coverage'
  },
  {
    value: 'item1',
    label: 'combination coverage'
  },
  {
    value: 'item2',
    label: 'internal coverage'
  },
  {
    value: 'item3',
    label: 'layer coverage'
  }
]
let second_value = ref('')

let third_options = [
  {
    value: 'item0',
    label: 'layer 0'
  },
  {
    value: 'item1',
    label: 'layer 1'
  },
  {
    value: 'item2',
    label: 'layer 2'
  },
  {
    value: 'item3',
    label: 'layer 3'
  },
  {
    value: 'item4',
    label: 'layer 4'
  },
  {
    value: 'item5',
    label: 'layer 5'
  },
  {
    value: 'item6',
    label: 'layer 6'
  },
  {
    value: 'item7',
    label: 'layer 7'
  },
  {
    value: 'item8',
    label: 'layer 8'
  },
]
let third_value = ref('')

let forth_options = [
  {
    value: 'detail',
    label: 'detail'
  },
  {
    value: 'abstract',
    label: 'abstract'
  }
]
let forth_value = ref('')

var route_name = ref('')

watch(forth_value, (newValue, oldValue) => {
  route_name.value = forth_value.value
})

// This starter template is using Vue 3 experimental <script setup> SFCs
// Check out https://github.com/vuejs/rfcs/blob/master/active-rfcs/0040-script-setup.md
</script>



<style>
img {
  margin: 0 auto;
  height: 20%;
  width: 20%;
}

a {
  /* color: #42b983; */
  color: #409eff;
  text-decoration: none;
}

.custom1 {
  border-color: #2c3e50;
  border-style: solid;
  border-width: 3px;
  border-radius: 8px;
  /* min-height: 36px; */
  height: 40px;
  line-height: 34px;
}

.custom2 {
  font-family: "Kaushan";
  font-size: larger;
}

.el-row {
  margin-bottom: 10px;
}
.el-col {
  border-radius: 4px;
}
.grid-content {
  color: #409eff;
  border-radius: 4px;
  /* min-height: 36px; */
  height: 40px;
  line-height: 40px;
}
.bg-purple {
  /* background: #d3dce6; */
  border-color: #409eff;
  border-style: solid;
  border-width: 3px;
}

#app {
  /* font-family: Avenir, Helvetica, Arial, sans-serif; */
  font-family: "Kaushan";
  font-size: larger;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 0px;
}
</style>
