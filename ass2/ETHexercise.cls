%

% ETHexercise.cls
%
% Copyright(c) 1995 by ETH Zuerich
% author: Niklaus Mannhart (mannhart@inf.ethz.ch), 31.3.1995
%
% This class file is based on Martin Wunderli's lehre.sty style file.
% Now, it compiles fine with LaTeX2e. Addtionally I changed the
% commands and gave them English names.
%
% Only English and German is implemented.
%
% History:
%   31.03.95 NM: creation
%    4.01.96 NM: updated some stuff (suggestions from Rolf)
%                  * new ETH header and the distinction
%                    between \serie, \seriepoint
%   28.05.96 NM: minor fixes (language problem is solved now)
%    4.03.97 NM: synchronised with suggestions from Rolf
%                  * new environments ex/sol
%   09.04.02 NK: changed layout to new ETH Logo
%
% ---------------------------------------------------------------------
% The file `ETHexercise.cls' is free software; you can redistribute
% it and/or modify it under the terms of the GNU General Public
% License as published by the Free Software Foundation version 2
% of the license.
%
% This class file is distributed in the hope that it will be usefull,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
% General Public License for more details.
% ---------------------------------------------------------------------
%
%
% options: german  loads the bable package automatically
%
% Example:
%
%   \documentclass[a4wide,11pt]{ETHexercise}
%
%   \lectureheader{Prof.~A.~Einstein}{G.~Globi}{Relativity Theory I}
%   \setserie{1}
%   \header{Serie \serie}{April 1st, 1912}{April 15, 1912}
%
%   \begin{question}{$E=mc^2$?}\label{..}
%      Given ....
%      \begin{subquestion}\label{...}
%         Assume that ...
%         \begin{subsubquestion}
%            Prove that ...
%         \end{subquestion}
%         other bla bla
%         \begin{subquestion}
%            Yet another prove...
%         \end{subquestion}
%      \end{subquestion}
%     ...
%     \begin{subquestion}
%     ...
%     \end{subquestion}
%   ...
%   \end{question}
%   ...
%   \end{document}
%
% option a4wide saves a lot of paper!
%
\NeedsTeXFormat{LaTeX2e} \ProvidesClass{ETHexercise}[1995/03/31]
%
% if the german option is defined then load the babel package and set
% the default language to german.
\RequirePackage{ifthen}
\RequirePackage{verbatim}

\ifx\pdfoutput\undefined
\relax
\else
\RequirePackage{thumbpdf}
\fi

\newboolean{language@set} \setboolean{language@set}{false}
\newboolean{a4wide@set} \setboolean{a4wide@set}{false}
\newboolean{@english} \setboolean{@english}{true}
\newboolean{@german} \setboolean{@german}{false}
\newboolean{ex@set} \setboolean{ex@set}{false}
\newboolean{sol@set} \setboolean{sol@set}{false}
%
\DeclareOption{german}{%
  \PassOptionsToPackage{german}{babel}
  \setboolean{language@set}{true}
  \setboolean{@german}{true}
  \setboolean{@english}{false}
}
\DeclareOption{a4wide}{%
  \setlength{\paperheight}{297mm}%
  \setlength{\paperwidth}{210mm}%
  \setboolean{a4wide@set}{true}
}
\DeclareOption{ex}{%
  \setboolean{ex@set}{true}
}
\DeclareOption{sol}{%
  \setboolean{sol@set}{true}
}
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{article}
}
\ProcessOptions
\LoadClass{article}
\ifthenelse{\boolean{language@set}}{% then load babel package
    \RequirePackage{babel}
}{}
\ifthenelse{\boolean{a4wide@set}}{% then set paper width
  \setlength{\marginparsep}{0mm}%
  \setlength{\marginparwidth}{0mm}%
  \setlength{\oddsidemargin}{3mm}%
  \setlength{\evensidemargin}{-5mm}%
  \setlength{\topmargin}{-15mm}%
  \setlength{\textheight}{250mm}%
  \setlength{\textwidth}{160mm}%
}{}
%
\ifx\pdfoutput\undefined
\usepackage[dvips]{graphicx}
\else
\usepackage[pdftex]{graphicx}
\fi
\input{ETHheaderpath.tex}
%
%
% define the words we use in this class file. Here you have to make
% changes if you are going to this file for a different language.
%
% BUG in latex2e, so we hard code it
%\iflanguage{german}{%
\ifthenelse{\boolean{@german}}{% then translate
  \newcommand{\Question}{Aufgabe}%
  }{% else English
  \newcommand{\Question}{Question}%
}
% define \ifex, \ifsol, \ifexsol
%
\newif\ifex\exfalse
\newif\ifsol\solfalse
\newif\ifexsol\exsolfalse
\ifthenelse{\boolean{sol@set}}{%
  \soltrue
  \ifthenelse{\boolean{ex@set}}{%
    \extrue\exsoltrue
  }{}%
}{%
  \extrue
}%
%
%
%
% global stuff: no parindent and parskip is set to 1ex
%
\parindent0cm
\parskip1.0ex plus 0.3ex minus 0.3ex
%
% \lectureheader{Prof}{Assistents}{title of the lecture}
%
% e.g: \lectureheader{Prof.~A.~Einstein}{R.~Theory, G.~Globbi}
%       {Relativity Theory}{Wintersemester Year}
%
\newcommand{\lectureheader}[4]{%
  \begin{minipage}{.3\textwidth}%
    \ifx\pdfoutput\undefined
    \strut\includegraphics[width=\textwidth]{\ETHheaderpath/ethlogo.eps}%
    \else
    \strut\includegraphics[width=\textwidth]{\ETHheaderpath/ethlogo.pdf}%
    \fi
  \end{minipage} \hfill%
  \raisebox{1.5mm}{%
    \begin{minipage}{0.69\textwidth}\sf\flushright%
        \textbf{\Huge #3}\mbox{\hspace{2mm}}\\#4\mbox{\hspace{2mm}}%
    \end{minipage}%
  }\\[-2mm]\hrule%
  \begin{minipage}[t]{0.5\textwidth}\sf\textit{#1} \end{minipage} \hfill%
  \begin{minipage}[t]{0.5\textwidth}\sf\flushright \textit{#2}\end{minipage}%
  \par%
}
%
% \header{exercise title}{handed out}{due to}
%
% \header{Serie 1}{April 1st, 1912}{April 15, 1912}
%
\newcommand{\header}[3]{%
  \vspace{-0.5em}
  \begin{center}\sf
    \textbf{\Large #1} \\[0.5ex]
    \textrm{\sf \normalsize #2} \\
    \textrm{} \\
    %\textrm{\sf Abgabe: \normalsize #3} \\
  \end{center}\par%
  \vspace{0.5em}
}
%
% \serie returns the serie number.
%
\newcommand{\serie}{not defined yet}
\newcommand{\seriepoint}{}
%
% \setserie sets the serie number.
%
\newcommand{\setserie}[1]{%
  \renewcommand{\serie}{#1}%
}
%
% \setseriepoint sets the serie number . question number
\newcommand{\setseriepoint}[1]{%
  \setserie{#1}
  \renewcommand{\seriepoint}{#1.}%
}%
%
% \sig{signature}
%
% \sig{AE 15 Apr 12}
\newcommand{\sig}[1]{%
  \vfill
  \hfill\emph{\footnotesize #1}
}
%
%
% question, subquestion and subsubquestions counter
%
\newcounter{questionC}
\newcounter{subquestionC}[questionC]
\newcounter{subsubquestionC}[subquestionC]
%
%
% save old label values because we set them back in the environments
% described below.
%
\newlength{\labelwidthbak}\setlength{\labelwidthbak}{\labelwidth}
\newlength{\leftmarginbak}\setlength{\leftmarginbak}{\leftmargin}
\newlength{\itemindentbak}\setlength{\itemindentbak}{\itemindent}
\newlength{\labelsepbak}\setlength{\labelsepbak}{\labelsep}
%
%
% \begin{question} .... \end{question}
%
\newenvironment{question}[1]{%
  \stepcounter{questionC} % advance counter by one
  \edef\@currentlabel{%
    \arabic{questionC}}%
  \begin{list}{\textbf{\large \Question~\seriepoint\arabic{questionC}: #1}}%
    {\labelwidth0pt \leftmargin0pt%
    \itemindent0pt \labelsep0pt}
    \item~\par%
    % restore old values
    \setlength{\labelwidth}{\labelwidthbak}%
    \setlength{\leftmargin}{\leftmarginbak}%
    \setlength{\itemindent}{\itemindentbak}%
    \setlength{\labelsep}{\labelsepbak}%
  }%
  {\end{list}}
%
%
% \begin{subquestion} ... \end{subquestion}
%
\newenvironment{subquestion}{%
  \stepcounter{subquestionC}
  \edef\@currentlabel{%
    \arabic{questionC}\alph{subquestionC}}%
  \begin{list}{\alph{subquestionC})}%
     {\settowidth{\labelwidth}{a) } \labelsep0.5em \topsep0pt
      \setlength{\leftmargin}{\labelwidth}%
      \addtolength{\leftmargin}{\labelsep}%
     }
     \item
     % reset values
     \setlength{\labelwidth}{\labelwidthbak}%
     \setlength{\leftmargin}{\leftmarginbak}%
     \setlength{\itemindent}{\itemindentbak}%
     \setlength{\labelsep}{\labelsepbak}%
  }%
  {\end{list}}
%
%
% \begin{subsubquestion} ... \end{subsubquestion}
%
\newenvironment{subsubquestion}{%
  \stepcounter{subsubquestionC}
  \edef\@currentlabel{%
    \arabic{questionC}.\alph{subquestionC}.\roman{subsubquestionC}}%
  \begin{list}{\roman{subsubquestionC})}%
    {\settowidth{\labelwidth}{viii) }%
    \labelsep0.5em \topsep0pt%
    \setlength{\leftmargin}{\labelwidth}%
    \addtolength{\leftmargin}{\labelsep}%
    }%
    \item
    % reset list values
    \setlength{\labelwidth}{\labelwidthbak}%
    \setlength{\leftmargin}{\leftmarginbak}%
    \setlength{\itemindent}{\itemindentbak}%
    \setlength{\labelsep}{\labelsepbak}%
  }%
  {\end{list}}
%
%
% \begin{(ex|sol)} ... \end{(ex|sol)}
%
\font\blackfont=cmbx10 scaled 400
\newcommand\strip[1]{%
  \leavevmode\xleaders\hbox{{\blackfont #1}}\hfill\strut%
  \par\nobreak\medskip\nobreak%
}

\newenvironment{ex}{%
  \ifexsol\strip{QUESTION}\fi%
  \ifex\let\next\relax\else\let\next\comment\fi\next%
}{%
  \ifex\par\else\endcomment\fi%
}%
\newenvironment{sol}{%
  \ifsol\strip{ANSWER}\fi%
  \em%
  \ifsol\let\next\relax\else\let\next\goodbreak\fi\next%
  \ifsol\let\next\relax\else\let\next\comment\fi\next%
}{%
  \ifsol\par\else\endcomment\fi%
  \ifsol\strip{ANSWER}\fi
  \ifsol\vspace{0.5cm}\fi
}%
%
%%%% That's it %%%%
%
