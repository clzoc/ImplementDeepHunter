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
      path: '/detail',
      component: () => import('/src/components/detail.vue')
    },
    {
      path: '/abstract',
      component: () => import('/src/components/abstract.vue'),
    }
  ]
})