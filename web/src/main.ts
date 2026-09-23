import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import './styles/base.css'
import 'uplot/dist/uPlot.min.css'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'lab',
      component: () => import('./pages/LabPage.vue'),
      meta: { title: '实验室' },
    },
    {
      path: '/exp/:id',
      name: 'exp',
      component: () => import('./pages/ExpLivePage.vue'),
      meta: { title: '训练实况' },
    },
    {
      path: '/exp/:id/inspect',
      name: 'inspect',
      component: () => import('./pages/InspectPage.vue'),
      meta: { title: 'AI 在想什么' },
    },
    {
      path: '/compare',
      name: 'compare',
      component: () => import('./pages/ComparePage.vue'),
      meta: { title: '对比' },
    },
    {
      path: '/play',
      name: 'play',
      component: () => import('./pages/PlayPage.vue'),
      meta: { title: '自己玩' },
    },
  ],
  scrollBehavior() {
    return { top: 0 }
  },
})

router.afterEach((to) => {
  const t = (to.meta.title as string) || '实验室'
  document.title = `${t} · 贪吃蛇 AI 训练实验室`
})

createApp(App).use(router).mount('#app')
