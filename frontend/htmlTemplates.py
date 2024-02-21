css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
 /* Chat containers */
.container {
  #border: 1px solid #dedede;
  #background-color: #efefef;
    background: rgba(0, 0, 0, .3);
  border-radius: 15px;
  max-width: 450px;
  padding: 10px;
  margin: 10px 0;
  color: rgba(255, 255, 255, .5);
  

}

/* Darker chat container */
.darker {
  #border-color: #ccc;
  #background-color: #e8f1f3;
  background: linear-gradient(120deg, #248A52, #257287);
  display: flex;
  max-width: 450px;
  margin-bottom: 10px;
  float: right;
  color: #fff;
  
}

/* Clear floats */
.container::after {
  content: "";
  clear: both;
  display: table;
}

/* Style images */
.container img {
  float: left;
  max-width: 60px;
  width: 100%;
  margin-right: 20px;
  border-radius: 50%;
}

/* Style the right image */
.container img.right {
  float: right;
  display: flex;
  max-width: 60px;
  width: 100%;
  margin-left: 20px;
  margin: 0px 0px 15px 15px;
}

/* Style time text */
.time-right {
  float: right;
  color: #aaa;
}

/* Style time text */
.time-left {
  float: left;
  color: #999;
} 
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

bot1_templete = '''
    <div class="msg-bubble">
        <div class="msg-info">
          <div class="msg-info-name">BOT</div>
          <div class="msg-info-time">12:45</div>
        </div>

        <div class="msg-text">
          {{MSG}}
        </div>
    </div>
'''



user_template = '''
<div class="chat-message user">  
    <div class="message">{{MSG}}</div>
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>  
</div>
'''

bot2_template = '''
<div class="container">
  <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Avatar">
  <p>{{MSG}}</p>
</div>
'''   

user3_template = '''
<div class="container darker">
  <p>{{MSG}}</p>
  <img  src="https://png.pngtree.com/png-clipart/20190924/original/pngtree-user-vector-avatar-png-image_4830521.jpg" alt='Avatar' class="right">
</div>
'''
