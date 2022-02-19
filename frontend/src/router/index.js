import { createRouter, createWebHistory } from 'vue-router'

export default createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      component: () => import('/src/components/HelloWorld.vue'),
      // props: { msg: "Wait for padding" }
    },
    {
      path: '/activationFeature',
      component: () => import('/src/components/activationFeature.vue')
    },
    {
      path: '/other',
      component: () => import('/src/components/other.vue'),
    },
    {
      path: '/neuronCoverage',
      component: () => import('/src/components/neuronCoverage.vue')
    },
    {
      path: '/internalCoverage',
      component: () => import('/src/components/internalCoverage.vue')
    },
    {
      path: '/layerTopCoverage',
      component: () => import('/src/components/layerTopCoverage.vue')
    },
    {
      path: '/combinationCoverage64',
      component: () => import('/src/components/combinationCoverage/combinationCoverage64.vue')
    },
    {
      path: '/combinationCoverage128',
      component: () => import('/src/components/combinationCoverage/combinationCoverage128.vue')
    },
    {
      path: '/combinationCoverage256',
      component: () => import('/src/components/combinationCoverage/combinationCoverage256.vue')
    },
    {
      path: '/combinationCoverage512',
      component: () => import('/src/components/combinationCoverage/combinationCoverage512.vue')
    },
    {
      path: '/combinationCoverage1024',
      component: () => import('/src/components/combinationCoverage/combinationCoverage1024.vue')
    },
    {
      path: '/combinationCoverage2048',
      component: () => import('/src/components/combinationCoverage/combinationCoverage2048.vue')
    },
  ]
})