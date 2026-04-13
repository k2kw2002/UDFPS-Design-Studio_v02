/**
 * drawMetricsCard.js - Skewness / MTF / Throughput 3 cards
 */
function drawMetricsCard(container, metrics) {
  var skewness = metrics.skewness;
  var mtf = metrics.mtf;
  var throughput = metrics.throughput;

  function grade(val, thresholds) {
    if (thresholds.higher) {
      return val > thresholds.good ? 'good' : val > thresholds.warn ? 'warn' : 'bad';
    }
    return val < thresholds.good ? 'good' : val < thresholds.warn ? 'warn' : 'bad';
  }

  function gradeLabel(cls) {
    return cls === 'good' ? 'Good' : cls === 'warn' ? 'Marginal' : 'Poor';
  }

  var cards = [
    { label: 'Skewness', value: skewness.toFixed(4),
      cls: grade(skewness, {good: 0.1, warn: 0.2, higher: false}) },
    { label: 'MTF (peak ratio)', value: mtf.toFixed(4),
      cls: grade(mtf, {good: 0.7, warn: 0.5, higher: true}) },
    { label: 'Throughput', value: throughput.toFixed(3),
      cls: grade(throughput, {good: 0.5, warn: 0.3, higher: true}) },
  ];

  container.innerHTML = '<div class="metrics-row">' + cards.map(function(c) {
    return '<div class="metric-card">' +
      '<div class="metric-label">' + c.label + '</div>' +
      '<div class="metric-value ' + c.cls + '">' + c.value + '</div>' +
      '<div class="metric-sub">' + gradeLabel(c.cls) + '</div>' +
      '</div>';
  }).join('') + '</div>';
}
