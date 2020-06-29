var path = require('../config/path.js');
var util = require('../util/util.js');
var http = require('http');
var https = require('https');
var fs = require('fs');

function execute(client_req, client_res) {
    util.logger.info('>>>>> EXECUTE([' + client_req.ip + '] ' + client_req.url + ') <<<<<');
    var proxy_url = client_req.url.replace('/', '/p_');
    util.logger.debug('proxy_url : ' + proxy_url);

    util.logger.debug('1');
    var ca = fs.readFileSync(path.ssl.ca).toString();
    util.logger.debug('2');
    var certificate = fs.readFileSync(path.ssl.cert).toString();
    util.logger.debug('3');
    var privateKey = fs.readFileSync(path.ssl.key).toString();
    util.logger.debug('4');
    var options = {
        hostname: path.proxy.host || "123.0.0.1",
        port: path.proxy.port || 8080,
        path: proxy_url,
        method: client_req.method,
        headers: client_req.headers
        //ca : ca

    };

    //   options.headers.Host = options.hostname + ":" + options.port;
    //   options.headers.Origin = "http://" + options.hostname + ":" + options.port;
    util.logger.debug('5 : ' + options['hostname'] +' '+ options['port']+' '+ options['path']+' '+ options['method']);
    var proxy = http.request(options, function (res) {
        util.logger.debug('10 : ' + res.statusCode);
        client_res.writeHead(res.statusCode, res.headers)
        util.logger.debug('11');
        res.pipe(client_res, {
            end: true
        });
    });
    util.logger.debug('6');
    proxy.on('error', function (err) {
        util.logger.debug('12');
        util.logger.error('ERROR - ' + err);
        client_res.send(util.makeResult('9999'));
    });
    util.logger.debug('3');
    client_req.pipe(proxy, {
        end: true
    });
    util.logger.debug('9');
}

module.exports = execute;

