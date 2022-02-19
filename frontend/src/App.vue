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
        <el-select
          popper-class="custom2"
          v-model="first_value"
          clearable
          placeholder="neural network"
        >
          <el-option
            v-for="item in first_options"
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
          v-model="second_value"
          clearable
          placeholder="display type"
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
      <div class="grid-content bg-purple">Select Coverage</div>
    </el-col>

    <el-col :span="2">
      <div class="custom1">
        <el-select
          :disabled="depend.valueOf()"
          popper-class="custom2"
          v-model="third_value"
          clearable
          placeholder="coverage kind"
        >
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
      <div class="grid-content bg-purple">Select Layer</div>
    </el-col>

    <el-col :span="2">
      <div class="custom1">
        <el-select
          :disabled="depend.valueOf()"
          popper-class="custom2"
          v-model="forth_value"
          clearable
          placeholder="layer index"
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
        <router-link
          :to="{ path: '/' + second_select.valueOf() + coverkind.valueOf() + correspondsize.valueOf(), query: { layer: layer_name.valueOf() } }"
        >Go</router-link>
      </div>
    </el-col>
  </el-row>

  <router-view></router-view>
  <!-- <router-view /> -->
</template>

<script setup>
// import HelloWorld from './components/HelloWorld.vue'
import "./assets/font/font.css"
import { reactive, ref, watch } from 'vue'

var correspondsize = ref('')
var arr64 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '15', '16', '17', '18', '19', '20', '24',
  '25', '26', '27', '28', '29']

var arr128 = ['33', '34', '35', '36', '37', '38', '44', '45', '46', '47', '48', '49', '53', '54', '55', '56', '57',
  '58', '62', '63', '64', '65', '66', '67']

var arr256 = ['10', '11', '12', '13', '14', '21', '22', '23', '30', '31', '32', '71', '72',
  '73', '74', '75', '76', '82', '83', '84', '85', '86', '87', '91', '92', '93',
  '94', '95', '96', '100', '101', '102', '103', '104', '105', '109', '110', '111', '112',
  '113', '114', '118', '119', '120', '121', '122', '123']

var arr512 = ['39', '40', '41', '42', '43', '50', '51', '52', '59', '60', '61', '68', '69',
  '70', '127', '128', '129', '130', '131', '132', '138', '139', '140', '141', '142', '143',
  '147', '148', '149', '150', '151', '152']

var arr1024 = ['77', '78', '79', '80', '81', '88', '89', '90', '97', '98', '99', '106', '107',
  '108', '115', '116', '117', '124', '125', '126']

var arr2048 = ['133', '134', '135', '136', '137', '144', '145', '146', '153', '154', '155', '156']

let first_options = [
  {
    value: 'item0',
    label: 'Resnet50'
  },
  // {
  //   value: 'item1',
  //   label: 'YoloV5'
  // }
]
let first_value = ref('')

let second_options = [
  {
    value: 'cover',
    label: 'cover'
  },
  {
    value: 'other',
    label: 'other'
  },
  {
    value: 'activationFeature',
    label: 'activationFeature'
  }
]

let second_value = ref('')
var second_select = ref('')
var depend = ref(true)
watch(second_value, (newValue, oldValue) => {
  second_select.value = second_value.value
  if (second_select.value == 'cover') {
    depend.value = false
    second_select.value = ''
  } else {
    depend.value = true
    coverkind.value = ''
  }
})


let third_options = [
  {
    value: 'neuronCoverage',
    label: 'neuronCoverage'
  },
  {
    value: 'combinationCoverage',
    label: 'combinationCoverage'
  },
  {
    value: 'internalCoverage',
    label: 'internalCoverage'
  },
  {
    value: 'layerTopCoverage',
    label: 'layerTopCoverage'
  }
]
let third_value = ref('')
var coverkind = ref('')
watch(third_value, (newValue, oldValue) => {
  coverkind.value = third_value.value
  if (coverkind.value == 'combinationCoverage') {
    if (arr64.indexOf(layer_name.value) == -1) {
      if (arr128.indexOf(layer_name.value) == -1) {
        if (arr256.indexOf(layer_name.value) == -1) {
          if (arr512.indexOf(layer_name.value) == -1) {
            if (arr1024.indexOf(layer_name.value) == -1) {
              if (arr2048.indexOf(layer_name.value) == -1) {
                correspondsize.value == ''
              }
              else {
                correspondsize.value == '2048'
              }
            }
            else {
              correspondsize.value = '1024'
            }
          }
          else {
            correspondsize.value = '512'
          }
        }
        else {
          correspondsize.value = '256'
        }
      }
      else {
        correspondsize.value = '128'
      }
    }
    else {
      correspondsize.value = '64'
    }
  }
  else {
    correspondsize.value = ''
  }
})

let forth_options = [
  {
    value: '0',
    label: '0'
  },
  {
    value: '1',
    label: '1'
  },
  {
    value: '2',
    label: '2'
  },
  {
    value: '3',
    label: '3'
  },
  {
    value: '4',
    label: '4'
  },
  {
    value: '5',
    label: '5'
  },
  {
    value: '6',
    label: '6'
  },
  {
    value: '7',
    label: '7'
  },
  {
    value: '8',
    label: '8'
  },
  {
    value: '9',
    label: '9'
  },
  {
    value: '10',
    label: '10'
  },
  {
    value: '11',
    label: '11'
  },
  {
    value: '12',
    label: '12'
  },
  {
    value: '13',
    label: '13'
  },
  {
    value: '14',
    label: '14'
  },
  {
    value: '15',
    label: '15'
  },
  {
    value: '16',
    label: '16'
  },
  {
    value: '17',
    label: '17'
  },
  {
    value: '18',
    label: '18'
  },
  {
    value: '19',
    label: '19'
  },
  {
    value: '20',
    label: '20'
  },
  {
    value: '21',
    label: '21'
  },
  {
    value: '22',
    label: '22'
  },
  {
    value: '23',
    label: '23'
  },
  {
    value: '24',
    label: '24'
  },
  {
    value: '25',
    label: '25'
  },
  {
    value: '26',
    label: '26'
  },
  {
    value: '27',
    label: '27'
  },
  {
    value: '28',
    label: '28'
  },
  {
    value: '29',
    label: '29'
  },
  {
    value: '30',
    label: '30'
  },
  {
    value: '31',
    label: '31'
  },
  {
    value: '32',
    label: '32'
  },
  {
    value: '33',
    label: '33'
  },
  {
    value: '34',
    label: '34'
  },
  {
    value: '35',
    label: '35'
  },
  {
    value: '36',
    label: '36'
  },
  {
    value: '37',
    label: '37'
  },
  {
    value: '38',
    label: '38'
  },
  {
    value: '39',
    label: '39'
  },
  {
    value: '40',
    label: '40'
  },
  {
    value: '41',
    label: '41'
  },
  {
    value: '42',
    label: '42'
  },
  {
    value: '43',
    label: '43'
  },
  {
    value: '44',
    label: '44'
  },
  {
    value: '45',
    label: '45'
  },
  {
    value: '46',
    label: '46'
  },
  {
    value: '47',
    label: '47'
  },
  {
    value: '48',
    label: '48'
  },
  {
    value: '49',
    label: '49'
  },
  {
    value: '50',
    label: '50'
  },
  {
    value: '51',
    label: '51'
  },
  {
    value: '52',
    label: '52'
  },
  {
    value: '53',
    label: '53'
  },
  {
    value: '54',
    label: '54'
  },
  {
    value: '55',
    label: '55'
  },
  {
    value: '56',
    label: '56'
  },
  {
    value: '57',
    label: '57'
  },
  {
    value: '58',
    label: '58'
  },
  {
    value: '59',
    label: '59'
  },
  {
    value: '60',
    label: '60'
  },
  {
    value: '61',
    label: '61'
  },
  {
    value: '62',
    label: '62'
  },
  {
    value: '63',
    label: '63'
  },
  {
    value: '64',
    label: '64'
  },
  {
    value: '65',
    label: '65'
  },
  {
    value: '66',
    label: '66'
  },
  {
    value: '67',
    label: '67'
  },
  {
    value: '68',
    label: '68'
  },
  {
    value: '69',
    label: '69'
  },
  {
    value: '70',
    label: '70'
  },
  {
    value: '71',
    label: '71'
  },
  {
    value: '72',
    label: '72'
  },
  {
    value: '73',
    label: '73'
  },
  {
    value: '74',
    label: '74'
  },
  {
    value: '75',
    label: '75'
  },
  {
    value: '76',
    label: '76'
  },
  {
    value: '77',
    label: '77'
  },
  {
    value: '78',
    label: '78'
  },
  {
    value: '79',
    label: '79'
  },
  {
    value: '80',
    label: '80'
  },
  {
    value: '81',
    label: '81'
  },
  {
    value: '82',
    label: '82'
  },
  {
    value: '83',
    label: '83'
  },
  {
    value: '84',
    label: '84'
  },
  {
    value: '85',
    label: '85'
  },
  {
    value: '86',
    label: '86'
  },
  {
    value: '87',
    label: '87'
  },
  {
    value: '88',
    label: '88'
  },
  {
    value: '89',
    label: '89'
  },
  {
    value: '90',
    label: '90'
  },
  {
    value: '91',
    label: '91'
  },
  {
    value: '92',
    label: '92'
  },
  {
    value: '93',
    label: '93'
  },
  {
    value: '94',
    label: '94'
  },
  {
    value: '95',
    label: '95'
  },
  {
    value: '96',
    label: '96'
  },
  {
    value: '97',
    label: '97'
  },
  {
    value: '98',
    label: '98'
  },
  {
    value: '99',
    label: '99'
  },
  {
    value: '100',
    label: '100'
  },
  {
    value: '101',
    label: '101'
  },
  {
    value: '102',
    label: '102'
  },
  {
    value: '103',
    label: '103'
  },
  {
    value: '104',
    label: '104'
  },
  {
    value: '105',
    label: '105'
  },
  {
    value: '106',
    label: '106'
  },
  {
    value: '107',
    label: '107'
  },
  {
    value: '108',
    label: '108'
  },
  {
    value: '109',
    label: '109'
  },
  {
    value: '110',
    label: '110'
  },
  {
    value: '111',
    label: '111'
  },
  {
    value: '112',
    label: '112'
  },
  {
    value: '113',
    label: '113'
  },
  {
    value: '114',
    label: '114'
  },
  {
    value: '115',
    label: '115'
  },
  {
    value: '116',
    label: '116'
  },
  {
    value: '117',
    label: '117'
  },
  {
    value: '118',
    label: '118'
  },
  {
    value: '119',
    label: '119'
  },
  {
    value: '120',
    label: '120'
  },
  {
    value: '121',
    label: '121'
  },
  {
    value: '122',
    label: '122'
  },
  {
    value: '123',
    label: '123'
  },
  {
    value: '124',
    label: '124'
  },
  {
    value: '125',
    label: '125'
  },
  {
    value: '126',
    label: '126'
  },
  {
    value: '127',
    label: '127'
  },
  {
    value: '128',
    label: '128'
  },
  {
    value: '129',
    label: '129'
  },
  {
    value: '130',
    label: '130'
  },
  {
    value: '131',
    label: '131'
  },
  {
    value: '132',
    label: '132'
  },
  {
    value: '133',
    label: '133'
  },
  {
    value: '134',
    label: '134'
  },
  {
    value: '135',
    label: '135'
  },
  {
    value: '136',
    label: '136'
  },
  {
    value: '137',
    label: '137'
  },
  {
    value: '138',
    label: '138'
  },
  {
    value: '139',
    label: '139'
  },
  {
    value: '140',
    label: '140'
  },
  {
    value: '141',
    label: '141'
  },
  {
    value: '142',
    label: '142'
  },
  {
    value: '143',
    label: '143'
  },
  {
    value: '144',
    label: '144'
  },
  {
    value: '145',
    label: '145'
  },
  {
    value: '146',
    label: '146'
  },
  {
    value: '147',
    label: '147'
  },
  {
    value: '148',
    label: '148'
  },
  {
    value: '149',
    label: '149'
  },
  {
    value: '150',
    label: '150'
  },
  {
    value: '151',
    label: '151'
  },
  {
    value: '152',
    label: '152'
  },
  {
    value: '153',
    label: '153'
  },
  {
    value: '154',
    label: '154'
  },
  {
    value: '155',
    label: '155'
  },
  {
    value: '156',
    label: '156'
  },
]
let forth_value = ref('')
var layer_name = ref('')
watch(forth_value, (newValue, oldValue) => {
  layer_name.value = forth_value.value
  if (coverkind.value == 'combinationCoverage') {
    if (arr64.indexOf(layer_name.value) == -1) {
      if (arr128.indexOf(layer_name.value) == -1) {
        if (arr256.indexOf(layer_name.value) == -1) {
          if (arr512.indexOf(layer_name.value) == -1) {
            if (arr1024.indexOf(layer_name.value) == -1) {
              if (arr2048.indexOf(layer_name.value) == -1) {
                correspondsize.value == ''
              }
              else {
                correspondsize.value == '2048'
              }
            }
            else {
              correspondsize.value = '1024'
            }
          }
          else {
            correspondsize.value = '512'
          }
        }
        else {
          correspondsize.value = '256'
        }
      }
      else {
        correspondsize.value = '128'
      }
    }
    else {
      correspondsize.value = '64'
    }
  }
  else {
    correspondsize.value = ''
  }
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
