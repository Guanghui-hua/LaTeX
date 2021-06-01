1. cls文件是什么？
LaTex中常见的文件格式有`.tex`, `.bib`, `.cls`, `.sty`, `.bbl`等，    
`.tex`文件也就是我们写文档内容的文件，  
`.bib`是使用bibligraphy方式导入参考文献时，写参考文献的文档，  
`.bbl`是其编译之后形成的文件，  
`.sty`是包文件，通常使用\usepackage导入，  
`.cls`是类文件，通过文档最前面的\documentclass命令导入~  

对于常见的article类，可以通过\usepackage[a4paper, 12pt]{article}的方式导入，同时定义了文档的尺寸为A4，默认字体大小为12pt。




cls文件开头都是
```latex
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{selftemplate}[template of Gray version 0.0]
\LoadClass{ctexart}
\RequirePackage{titlesec}
```
其中第一行指定了LaTeX编译器的版本需求。告诉了编译器这个宏包或者说类文件是属于哪个版本的。    
本文中宏包和类都指.cls文件。  

第二行指出了这个宏包的版本修改日期，**日期格式必须是代码中所示的那样**。  
\ProvidesClass命令的第一个参数就是cls文件的名字，两个必须一致；  
第二个参数是可选的，如果给出了第二个参数，则编译器会检查文档所引用的类文件，如果引用的不是最新版本会给出警告，但是一般文档中也不会指出具体的日期，所以作用不大。   

第三行给出了该类所继承的父类，一般宏包都是依托于其他宏包编写的。这里给出依托的其他类名，这样我们没有设置格式的其他文章结构都会使用父类默认的格式。  

第四行\RequirePackage命令指出该类需要使用的其他宏包，这个宏包和第三行的宏包不太一样，这里的宏包提供了设置格式的功能，第三行的宏包一般是给出了各种设置的格式，如chapter、section等的格式。此外，\RequirePackage保证引用的宏包只被引入一次，类似于C++中的#pragma once。  

