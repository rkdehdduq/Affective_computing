function checkLogin() {
    var id = document.getElementById("id").value;
    var pw = document.getElementById("pw").value;
  
    // JSON 파일에서 로그인 정보 가져오기
    fetch('login_info.json')
      .then(response => response.json())
      .then(data => {
        var storedId = data.username;
        var storedPw = data.password;
  
        // 입력한 로그인 정보와 저장된 정보 비교
        if (id === storedId && pw === storedPw) {
          alert("로그인 성공");
        } else {
          alert("로그인 실패");
        }
      })
      .catch(error => {
        console.error('로그인 정보를 가져올 수 없습니다.', error);
      });
  }
  