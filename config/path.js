const _pathBase = 'C:/Users/pyk12/Desktop/git/ui/poc_koreauniv_ui/koreaUnivProject_Was';
module.exports = {
		root: '/',
		port : 8080,
		//port: 8443, // 생략할 경우 기본 8443 사용
		logs: _pathBase + '/logs',
		proxy : {
			host : '15.164.138.33', // 생략할 경우 123.0.0.1
			port : 8082 //생략할 경우 8080
		}
		
};
