;
var https = require('http');
var express = require('express');
var session = require('express-session');
var bodyParser = require('body-parser')
var path = require('./config/path.js');
var util = require('./util/util.js');
var ipaddr = require('ipaddr.js');
var fs = require('fs');
var helmet = require('helmet');
var cors = require('cors');

var app = express();

app.set('port', path.port || 8888);
app.use(cors());

// app.use(bodyParser.urlencoded({
// 	extended: true
// }));

// app.use(session({
// 	secret: '@#@$S-BUILDER-PROXY#@$#$',
// 	resave: false,
// 	saveUninitialized: true,
// 	cookie: {
// 		maxAge: 1000 * 60 * 5 // 쿠키 유효기간 5분
// 	}
// }));

app.use(express.static(__dirname + '/public'));

app.get('/', function (req, res) {
	util.logger.info('/// ');
	var user_agent = req.headers['user-agent'];
	if (check_mobile_device(user_agent)) {
		res.sendFile(__dirname + '/public/index.html');
	}
	else {
		// desktop용 로그인 화면
		res.sendFile(__dirname + '/public/index.html');
	}
});

app.get('/index.html', function (req, res) {
	util.logger.info('///  index.html');
	var user_agent = req.headers['user-agent'];
	if (check_mobile_device(user_agent)) {
		res.sendFile(__dirname + '/public/index.html');
	}
	else {
		// desktop용 로그인 화면
		res.sendFile(__dirname + '/public/index.html');
	}
});





// var api_proxy = proxy({
// 	target: "http://" + path.proxy.host + ":" + path.proxy.port,
// 	pathRewrite: {
// 		'/login.do': '/p_login.do',
// 		'/logout.do': '/p_logout.do',
// 		'/download_project_list.do': '/p_download_project_list.do',
// 		'/download_project_build_type_list.do': '/p_download_project_build_type_list.do',
// 		'/download_app_list.do': '/p_download_app_list.do',
// 		'/download_app_apk.do': '/p_download_app_apk.do',
// 		'/download_app_ipa.do': '/p_download_app_ipa.do',
// 		'/download_app_plist_url.do': '/p_download_app_plist_url.do',
// 		'/download_list.do': '/p_download_list.do',
// 		'/cert_insert.do': '/p_cert_insert.do',
// 		'/provision_insert.do': '/p_provision_insert.do',
// 		'/sdk_insert.do': '/p_sdk_insert.do',
// 	},
// });


// app.post('/login.do', api_proxy);
// app.post('/logout.do', api_proxy);
// app.post('/download_project_list.do', api_proxy);
// app.post('/download_project_build_type_list.do', api_proxy);
// app.post('/download_app_list.do', api_proxy);
// app.get('/download_app_apk.do', api_proxy);
// app.get('/download_app_ipa.do', api_proxy);
// app.post('/download_app_plist_url.do', api_proxy);

// app.get('/plist/*.do', api_proxy);
app.post('/send_py.do', require('./api/send_py.js'));


// app.post('/download_list.do', api_proxy);
// app.post('/cert_insert.do', api_proxy);
// app.post('/provision_insert.do', api_proxy);
// app.post('/sdk_insert.do', api_proxy);


// login/logout
// app.post('/login.do', require('./api/interceptor.js'), require('./api/proxy.js'));
// app.post('/logout.do', require('./api/interceptor.js'), require('./api/proxy.js'));

// app.post('/account_session.do', require('./api/interceptor.js'), require('./api/proxy.js'));
// app.post('/download_project_list.do', require('./api/interceptor.js'), require('./api/proxy.js'));
// app.post('/download_project_build_type_list.do', require('./api/interceptor.js'), require('./api/proxy.js'));
// app.post('/download_app_list.do',require('./api/interceptor.js'), require( './api/proxy.js'));
// app.get('/download_app_apk.do', require('./api/interceptor.js'), require('./api/proxy.js'));
// app.get('/download_app_ipa.do', require('./api/interceptor.js'), require('./api/proxy.js'));
// app.post('/download_app_plist_url.do', require('./api/interceptor.js'), require('./api/proxy.js'));
// app.post('/download_list.do', require('./api/interceptor.js'), require('./api/proxy.js'));


// app.get('/plist/*.do', require('./api/interceptor.js'), require('./api/proxy_bypass.js'));
// app.get('/ipa/*.do', require('./api/interceptor.js'), require('./api/proxy_bypass.js'));


// https.createServer(credentials, app).listen(app.get('port'), function () {
// 	util.logger.info('S-Builder proxy server listening on port ' + app.get('port'));
// });

app.listen(app.get('port'), function () {
	util.logger.info('Best For you listening on port ' + app.get('port'));
});

function check_mobile_device(user_agent) {

	if (user_agent.match(/Android/i) != null)
		return true;


	if (user_agent.match(/iPhone|iPad|iPod/i) != null)
		return true;


	return false;
}