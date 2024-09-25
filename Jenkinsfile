pipeline {
    agent none
    
    stages {
        stage('Determine Node') {
            steps {
                script {
                    // Set the NODE_LABEL based on the branch name
                    if (env.BRANCH_NAME == 'dev/main') {
                        env.NODE_LABEL = 'DorisCT-dev'
                        env.GITHUB_CREDENTIALS = 'GitHub-Doris-CT-Dev' // SSH key for Dev
                    } else if (env.BRANCH_NAME == 'staging/main') {
                        env.NODE_LABEL = 'DorisCT-staging'
                        env.GITHUB_CREDENTIALS = 'GitHub-Doris-CT-Staging' // SSH key for Staging
                    } else if (env.BRANCH_NAME == 'prod/main') {
                        env.NODE_LABEL = 'DorisCT-prod'
                        env.GITHUB_CREDENTIALS = 'GitHub-Doris-CT-Prod' // SSH key for prod
                    } else {
                        error "No node label defined for branch: ${env.BRANCH_NAME}"
                    }
                    echo "Node label set to: ${env.NODE_LABEL}"
                    echo "Using GitHub credentials: ${env.GITHUB_CREDENTIALS}"
                }
            }
        }

        stage("Checkout Code") {
            agent {
                label "${env.NODE_LABEL}"
            }
            steps {
                checkout scmGit(branches: [[name: "${env.BRANCH_NAME}"]], extensions: [], userRemoteConfigs: [[credentialsId: "${env.GITHUB_CREDENTIALS}", url: 'git@github.com:surya-bhosale/dorisclinicaltrials.git']])
            }
        }

        stage("Clean Up") {
            agent {
                label "${env.NODE_LABEL}"
            }
            steps {
                sh "sudo docker system prune -f --all"        
            }
        }

        stage("Build & Deploy") {
            agent {
                label "${env.NODE_LABEL}"
            }
            steps {
                script {
                    // Custom build and deploy logic per branch
                    if (env.BRANCH_NAME == 'dev/main') {
                        sh "sudo docker compose -f docker-compose-dev.yml up --build --force-recreate -d"

                    } else if (env.BRANCH_NAME == 'staging/main') {
                        sh "sudo docker compose -f docker-compose-dev.yml up --build --force-recreate -d"
                        
                    } else if (env.BRANCH_NAME == 'prod/main') {
                        sh "sudo docker compose -f docker-compose-dev.yml up --build --force-recreate -d"
                        
                    }
                }
            }
        }
    }
    post { 
        always { 
            node(env.NODE_LABEL) 
            {
            script { 
                        def jobName = env.JOB_NAME.replaceAll('%2F', '/')
                        def buildNumber = env.BUILD_NUMBER 
                        def pipelineStatus = currentBuild.result ?: 'UNKNOWN' 
                        def bannerColor = pipelineStatus.toUpperCase() == 'SUCCESS' ? 'green' : 'red'
                        def changes = getGitChanges()
                        
        
                        def body = """<html> 
                                <body> 
                                    <div style="border: 4px solid ${bannerColor}; padding: 10px;"> 
                                        <h2>${jobName} - Build ${buildNumber}</h2> 
                                        <div style="background-color: ${bannerColor}; padding: 10px;"> 
                                            <h3 style="color: white;">Pipeline Status: ${pipelineStatus.toUpperCase()}</h3> 	
                                        </div>
                                        <p>Git Commit changes for the build: ${changes}
                                        Check the <a href="${BUILD_URL}">console output</a>.</p> 
                                    </div> 
                                </body> 
                            </html>""" 
                        emailext ( 
                            subject: "${jobName} - Build ${buildNumber} - ${pipelineStatus.toUpperCase()}", 
                            body: body, 
                            to: 'surya@dorisjv.com,shekhar@dorisjv.com,pavan@dorisjv.com,anup@dorisjv.com,hrishabh.mandhan@pyrack.com',
                            replyTo: 'hrishabh.mandhan@pyrack.com',
                            mimeType: 'text/html', 
                            
                        ) 
                    } 
                }
        }
    }
}
def getGitChanges() {
    def gitLog = sh(returnStdout: true, script: 'git log -1 --pretty=format:"%h by %an on %ad: %s" --date=iso').trim()
    def changes = "<li>${gitLog}</li>"
    return changes
}
