var path = require('../config/path.js');
var util = require('../util/util.js');
var http = require('http');
var https = require('https');
var fs = require('fs');

function execute(client_req, client_res) {
    util.logger.info('>>>>> EXECUTE(['+client_req.ip + '] ' + client_req.url + ') <<<<<');

    
    // res = requests.post("http:/123.0.0.1:5000", data=data)

  
        // var proxy_url = client_req.url;
    // util.logger.debug('proxy_url : ' + proxy_url);
  
    var options = {
        hostname: "123.0.0.1",
        port: 5000,
        path: '/send_py',
        method: client_req.method,
        headers: client_req.headers
    };
    
    var proxy = http.request(options, function (res) {
        alert(client_res.statusCode)
        // client_res.writeHead(res.statusCode, res.headers)
    
    });
    proxy.on('error', function (err) {
        util.logger.error('ERROR - ' + err);
        // client_res.send(util.makeResult('9999'));
    });

    // client_req.pipe(proxy, {
    //     end: true
    // });
}

module.exports = execute;

