/* Examples */
(function($) {
	/*
	 * Example 1:
	 *
	 * - no animation
	 * - custom gradient
	 *
	 * By the way - you may specify more than 2 colors for the gradient
	 */
	/*
	 * Example 2:
	 *
	 * - default gradient
	 * - listening to `circle-animation-progress` event and display the animation progress: from 0 to 100%
	 */
	$('.circle.circle01').circleProgress({
		//animation: false,
		//value: 1,
		fill: {gradient: [['#2490f3', .2], ['#54cbdf', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});

	$('.circle.circle02').circleProgress({
		//animation: false,
		//value: .25,
		fill: {gradient: [['#5461e3', .2], ['#2691f3', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});
	
	$('.circle.circle03').circleProgress({
		//animation: false,
		value: 1,
		fill: {gradient: [['#b425f3', .2], ['#5c4ee3', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});

	$('.circle.circle04').circleProgress({
		value: 1,
		//animation: false,
		value: 1,
		fill: {gradient: [['#e04f92', .2], ['#b523f0', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});
	
	$('.circle.circle05').circleProgress({
		value: 1,
		//animation: false,
		fill: {gradient: [['#f0a226', .2], ['#e34e8e', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});
	$('.circle.circle06').circleProgress({
		value: 1,
		//animation: false,
		fill: {gradient: [['#f3a326', .2], ['#e2d94c', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});
	$('.circle.circle03').circleProgress({
		value: 1,
		//animation: false,
		fill: {gradient: [['#89c83b', .2], ['#dbd84d', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});
	$('.circle.circle08').circleProgress({
		value: 1,
		//animation: false,
		fill: {gradient: [['#02b331', .2], ['#83ca3d', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});
	$('.circle.circle09').circleProgress({
		value: 1,
		//animation: false,
		fill: {gradient: [['#2ba5ce', .2], ['#02b233', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});
	$('.circle.circle10').circleProgress({
		value: 0, // 실행시 value: 1,
		animation: false, // 실행시 animation: false, 삭제
		fill: {gradient: [['#29cdcf', .2], ['#2aa5ce', .6]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<i>%</i>');
	});
	
	// load progress 2019-11-06 수정
	$('.circle.circle-load').circleProgress({
		//value: 1,
		size:124,
		animation: { duration: 5000 }, // 속도
		fill: {gradient: [['#fff', 1], ['#fff', 1]]},
		startAngle: -Math.PI / 4 * 2,
		lineCap: 'round',
	}).on('circle-animation-progress', function(event, progress) {
		$(this).find('strong').html(Math.round(100 * progress) + '<span>%</span>');
		$(this).find('.txt em').html(Math.round(1000 * progress));
	});
	
})(jQuery);
