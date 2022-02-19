<template>
  <el-container>
    <el-header>This is a project about domain specific coverage of deep learning models based on visualization techniques</el-header>
    <el-container>
      <el-aside>
        <img alt="scheme" src="../assets/scheme.jpg" style="width: 100%; height: 900px;" />
      </el-aside>
      <el-container>
        <el-main>
          <!-- <h1>{{ msg }}</h1> -->
        </el-main>
        <el-footer></el-footer>
      </el-container>
    </el-container>
  </el-container>

  <!-- <p>
    <a href="https://vitejs.dev/guide/features.html" target="_blank">Vite Documentation</a>
    |
    <a href="https://v3.vuejs.org/" target="_blank">Vue 3 Documentation</a>
  </p>

  <button type="button" @click="state.count++">count is: {{ state.count }}</button>
  <p>
    Edit
    <code>components/HelloWorld.vue</code> to test hot module replacement.
  </p>-->
</template>

<script setup>
// import { defineProps, reactive } from 'vue'

// defineProps({
//   msg: String
// })

// const state = reactive({ count: 0 })
</script>

<style scoped>
a {
  margin: 0 auto;
  color: #42b983;
}

.el-header {
  /* background-color: #b3c0d1; */
  /* background-color: #409eff; */
  /* color: var(--el-text-color-primary); */
  color: #2c3e50;
  text-align: center;
  line-height: 50px;
  height: 60px;
  border-radius: 4px;
  border-color: #409eff;
  border-style: solid;
  border-width: 3px;
  margin-bottom: 10px;
}

/* body > .el-container {
  margin-bottom: 40px;
} */

.el-footer {
  /* background-color: #b3c0d1; */
  /* background-color: #409eff; */
  /* color: var(--el-text-color-primary); */
  color: #2c3e50;
  text-align: center;
  line-height: 400px;
  height: 460px;
  border-radius: 4px;
  border-color: #409eff;
  border-style: solid;
  border-width: 3px;
  margin-top: 10px;
  margin-left: 10px;
}

.el-aside {
  /* background-color: #409eff; */
  /* color: var(--el-text-color-primary); */
  color: #2c3e50;
  text-align: center;
  line-height: 800px;
  width: 1603px;
  height: 902.8px;
  border-radius: 4px;
  border-color: #409eff;
  border-style: solid;
  border-width: 3px;
}

.el-main {
  /* background-color: #409eff; */
  /* color: var(--el-text-color-primary); */
  color: #2c3e50;
  text-align: center;
  line-height: 300px;
  border-radius: 4px;
  border-color: #409eff;
  border-style: solid;
  border-width: 3px;
  margin-left: 10px;
}

/* .el-container {
  margin-left: 10px;
} */

/* .el-container:nth-child(5) .el-aside,
.el-container:nth-child(6) .el-aside {
  line-height: 260px;
}

.el-container:nth-child(7) .el-aside {
  line-height: 320px;
} */
</style>
