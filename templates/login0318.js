const fs = require('fs');

// 로그인 정보 저장 함수
function saveLoginInfo(username, password) {
  const loginInfo = {
    username: username,
    password: password
  };

  fs.writeFileSync('login_info.json', JSON.stringify(loginInfo));
  console.log('로그인 정보 저장 완료');
}

// 로그인 정보 저장 예시
saveLoginInfo('일취월장', '1111');
