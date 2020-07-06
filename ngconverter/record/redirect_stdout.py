import os, sys, warnings


class RedirectStdout:
    def __init__(self):
        self.content = ''
        self.savedStdout = sys.stdout
        self.savedStderr = sys.stderr
        self.fileObj, self.nulObj = None, None
        def customwarn(message, category, filename, lineno, file=None, line=None):
            sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
        warnings.showwarning = customwarn

    #外部的print语句将执行本write()方法，并由当前sys.stdout输出
    def write(self, outStr):
        #self.content.append(outStr)
        self.content += outStr

    def toCons(self):  #标准输出重定向至控制台
        sys.stdout = self.savedStdout #sys.__stdout__
        sys.stderr = self.savedStderr #sys.__stdout__

    def toFile(self, file='out.txt'):  #标准输出重定向至文件
        self.fileObj = open(file, 'a+', 1) #改为行缓冲
        sys.stdout = self.fileObj
        sys.stderr = self.fileObj
    
    def toMute(self):  #抑制输出
        self.nulObj = open(os.devnull, 'w')
        sys.stdout = self.nulObj
        sys.stderr = self.nulObj
        
    def restore(self):
        self.content = ''
        if self.fileObj != None and self.fileObj.closed != True:
            self.fileObj.close()
        if self.nulObj != None and self.nulObj.closed != True:
            self.nulObj.close()
        sys.stdout = self.savedStdout #sys.__stdout__
        sys.stderr = self.savedStderr #sys.__stdout__
  
# Redirect Object
redirObj = RedirectStdout()
sys.stdout = redirObj

def direct_to_file(task):
    # Redirect log
    output_file_path = os.path.join(task.task_dir, "stdout.log")
    print("输出被转入到%s." % output_file_path)
    redirObj.toFile(output_file_path)

def direct_to_console():
    # Redirect log
    redirObj.toCons()

def close_redirection():
    # close redirection
    redirObj.restore()
