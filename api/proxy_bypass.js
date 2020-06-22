var path = require('../config/path.js');
var util = require('../util/util.js');
var http = require('http');
var https = require('https');
var fs = require('fs');

function execute(client_req, client_res) {
    util.logger.info('>>>>> EXECUTE(['+client_req.ip + '] ' + client_req.url + ') <<<<<');
    var proxy_url = client_req.url;
    util.logger.debug('proxy_url : ' + proxy_url);


    var ca = fs.readFileSync(path.ssl.ca).toString();
    

    var options = {
        hostname: path.proxy.host || "123.0.0.1",
        port: path.proxy.port || 8080,
        path: proxy_url,
        method: client_req.method,
        headers: client_req.headers
        //ca: ca
    };
    

    //   options.headers.Host = options.hostname + ":" + options.port;
    //   options.headers.Origin = "http://" + options.hostname + ":" + options.port;

    var proxy = http.request(options, function (res) {
        client_res.writeHead(res.statusCode, res.headers)
        res.pipe(client_res, {
            end: true
        });
    });
    proxy.on('error', function (err) {
        util.logger.error('ERROR - ' + err);
        client_res.send(util.makeResult('9999'));
    });

    client_req.pipe(proxy, {
        end: true
    });
}

module.exports = execute;

