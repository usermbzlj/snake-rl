<script setup lang="ts">
import { computed } from 'vue'
import UChart, { type ChartMarker } from '@/components/UChart.vue'
import type { Algo, MetricRow } from '@/api'
import { metricSeries, metricXs } from '@/utils/metrics'

const props = defineProps<{
  metrics: MetricRow[]
  markers: ChartMarker[]
  algo: Algo
}>()

const xs = computed(() => metricXs(props.metrics))
</script>

<template>
  <section class="charts panel">
    <h2 class="panel-title">训练曲线</h2>
    <div class="chart-grid">
      <UChart
        title="得分"
        hint="平均得分：最近一批局里，AI 大概能吃几颗豆。往上爬就说明在进步。"
        :x="xs"
        :series="[
          { label: '平均得分', data: metricSeries(metrics, 'score_mean') },
          { label: '最高得分', data: metricSeries(metrics, 'score_max'), color: '#f0c35a' },
        ]"
        :markers="markers"
        :height="220"
      />
      <UChart
        title="评估得分"
        hint="隔一会儿用「认真模式」打几局的成绩，比训练均分更诚实。"
        :x="xs"
        :series="[{ label: 'eval 均分', data: metricSeries(metrics, 'eval_score_mean'), color: '#6db3f2' }]"
        :markers="markers"
        :height="220"
      />
      <UChart
        title="死亡原因"
        hint="三类死法各占多少。撞自己多 → 还不会绕路；饿死多 → 找豆太慢。"
        :x="xs"
        :series="[
          { label: '撞墙', data: metricSeries(metrics, 'death_wall'), color: '#f07178' },
          { label: '撞自己', data: metricSeries(metrics, 'death_self'), color: '#f0c35a' },
          { label: '饿死', data: metricSeries(metrics, 'death_starve'), color: '#b794f6' },
        ]"
        :markers="markers"
        :height="220"
      />
      <UChart
        v-if="algo === 'ppo'"
        title="熵（探索欲）"
        hint="熵高＝更爱乱试；变低＝开始有固定打法。太低太早可能学偏了。"
        :x="xs"
        :series="[{ label: '熵', data: metricSeries(metrics, 'entropy') }]"
        :markers="markers"
        :height="220"
      />
      <UChart
        v-else
        title="探索率 ε"
        hint="ε 高时更爱随机试；慢慢降下来表示开始相信学到的经验。"
        :x="xs"
        :series="[{ label: 'ε', data: metricSeries(metrics, 'epsilon'), color: '#f0c35a' }]"
        :markers="markers"
        :height="220"
      />
      <UChart
        title="损失"
        hint="训练误差。整体往下走就好；偶尔跳一下通常正常。"
        :x="xs"
        :series="
          algo === 'ppo'
            ? [
                { label: '策略', data: metricSeries(metrics, 'loss_policy') },
                { label: '价值', data: metricSeries(metrics, 'loss_value'), color: '#6db3f2' },
              ]
            : [{ label: 'Q 损失', data: metricSeries(metrics, 'loss_q'), color: '#f07178' }]
        "
        :markers="markers"
        :height="220"
      />
    </div>
  </section>
</template>

<style scoped>
.charts {
  margin-top: 16px;
}
.chart-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}
@media (max-width: 1200px) {
  .chart-grid {
    grid-template-columns: 1fr 1fr;
  }
}
@media (max-width: 720px) {
  .chart-grid {
    grid-template-columns: 1fr;
  }
}
</style>
