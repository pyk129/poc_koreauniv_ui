var util = require('../util/util.js');

function execute(req, res, next) {
    util.logger.info('>>>>> INTERCEPTOR(['+req.ip + '] ' + req.url + ') <<<<<');
    next();
}

module.exports = execute;

