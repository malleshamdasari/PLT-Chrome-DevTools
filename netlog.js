var Chrome = require('chrome-remote-interface');

Chrome(function (chrome) {
    with (chrome) {
        Network.enable();
        Page.enable();
        // log all requests before they are sent
        Network.requestWillBeSent(function (params) {
            console.log(params.request);
        });
        // once page loads, navigate to another page
        once('ready', function () {
            Page.navigate({'url': 'http://www.zdnet.com'});
        });
    }
 });
