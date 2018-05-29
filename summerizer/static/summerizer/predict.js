$(function() {

  function isDisableGenerate(isDisable) {
    var $generate = $('#generate');

    if (isDisable) {
      $generate.addClass('loading');
      $generate.attr('disabled', true);
    } else {
      $generate.removeClass('loading');
      $generate.attr('disabled', false);
    }
  }
  
  $('#generate').click(function (e) {
    e.preventDefault();
    var body_text = $('#issue_body').val();
    isDisableGenerate(true);

    // predict api
    var url = 'http://localhost:8000/api/generate-title/?body=';
    url += encodeURI(body_text);
    console.log(url)

    $.ajax({
      url: url,
      type: 'GET',
      success: function (data) {
        console.log('ok');
        console.log(data);

        $('#generated_title').text(data.generated_title);
        isDisableGenerate(false);
      },
      error: function (err) {
        console.log('error');
        console.warn(err);
        isDisableGenerate(false);
      }
    })
  })

  $('#clear').click(function (e) {
    e.preventDefault();
    $('#generated-title').text('');
    $('#issue_body').val('');
  })

});